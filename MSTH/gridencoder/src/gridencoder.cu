#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be a bool tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
 __device__ inline at::Half atomicAdd(at::Half *address, at::Half val) {
  // requires CUDA >= 10 and ARCH >= 70
  // this is very slow compared to float or __half2, never use it.
  //return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename T, typename T2>
__host__ __device__ inline T clamp(const T v, const T2 lo, const T2 hi) {
  return min(max(v, lo), hi);
}

template <typename T>
__device__ inline T smoothstep(T val) {
	return val*val*(3.0f - 2.0f * val);
}

template <typename T>
__device__ inline T smoothstep_derivative(T val) {
	return 6*val*(1.0f - val);
}


template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
    
    // coherent type of hashing
    constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}


template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const uint32_t gridtype, const bool align_corners, const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += pos_grid[d] * stride;
        stride *= align_corners ? resolution: (resolution + 1);
    }

    // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
    // gridtype: 0 == hash, 1 == tiled
    if (gridtype == 0 && stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

    return (index % hashmap_size) * C + ch;
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    scalar_t * __restrict__ dy_dx,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate (always use float for precision!)
    float pos[D];
    float pos_deriv[D]; 
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos_deriv[d] = smoothstep_derivative(pos[d]);
            pos[d] = smoothstep(pos[d]);
        } else {
            pos_deriv[d] = 1.0f; // linear deriv is default to 1
        }

    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // interpolate
    scalar_t results[C] = {0}; // temp results in register

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }

        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }    

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

    // prepare dy_dx
    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (dy_dx) {

        dy_dx += b * D * L * C + level * D * C; // B L D C

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {

            scalar_t results_grad[C] = {0};

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale;
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }

                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);
                pos_grid_local[gd] = pos_grid[gd] + 1;
                uint32_t index_right = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_deriv[gd];
                }
            }

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                dy_dx[gd * C + ch] = results_grad[ch];
            }
        }
    }
}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos[d] = smoothstep(pos[d]);
        }
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
            }
        }
    }    
}


template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,  
    scalar_t * __restrict__ grad_inputs, 
    uint32_t B, uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= B * D) return;

    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    dy_dx += b * L * D * C;

    scalar_t result = 0;
    
    # pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }

    grad_inputs[t] = result;
}


template <typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, scalar_t *dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 2: kernel_grid<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 4: kernel_grid<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 8: kernel_grid<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [L, B, C], float (L first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
// dy_dx: [B, L * D * C]
template <typename scalar_t>
void grid_encode_forward_cuda(const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, scalar_t *dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    switch (D) {
        case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 4: kernel_grid_wrapper<scalar_t, 4>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 5: kernel_grid_wrapper<scalar_t, 5>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(const scalar_t *grad, const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 1: 
            kernel_grid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp); 
            if (dy_dx) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 2: 
            kernel_grid_backward<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
            if (dy_dx) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4: 
            kernel_grid_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
            if (dy_dx) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 8: 
            kernel_grid_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
            if (dy_dx) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution
template <typename scalar_t>
void grid_encode_backward_cuda(const scalar_t *grad, const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    switch (D) {
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 4: kernel_grid_backward_wrapper<scalar_t, 4>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 5: kernel_grid_backward_wrapper<scalar_t, 5>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}



void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "grid_encode_forward", ([&] {
        grid_encode_forward_cuda<scalar_t>(inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, S, H, dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, gridtype, align_corners, interp);
    }));
}

void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    // CHECK_CUDA(dy_dx);
    // CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    // CHECK_CONTIGUOUS(dy_dx);
    // CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "grid_encode_backward", ([&] {
        grid_encode_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, S, H, dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, grad_inputs.has_value() ? grad_inputs.value().data_ptr<scalar_t>() : nullptr, gridtype, align_corners, interp);
    }));
    
}


template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grad_tv(
    const scalar_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid, 
    scalar_t * __restrict__ grad, 
    const int * __restrict__ offsets, 
    const float weight,
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const uint32_t gridtype,
    const bool align_corners
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    inputs += b * D;
    grid += (uint32_t)offsets[level] * C;
    grad += (uint32_t)offsets[level] * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, do nothing
    if (flag_oob) return;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D]; // [0, resolution]

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        // pos[d] -= (float)pos_grid[d]; // not used
    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // total variation on pos_grid
    scalar_t results[C] = {0}; // temp results in register
    scalar_t idelta[C] = {0};

    uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

    scalar_t w = weight / (2 * D);

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {

        uint32_t cur_d = pos_grid[d];
        scalar_t grad_val;

        // right side
        if (cur_d < resolution) {
            pos_grid[d] = cur_d + 1;
            uint32_t index_right = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // results[ch] += w * clamp(grid[index + ch] - grid[index_right + ch], -1.0f, 1.0f);
                grad_val = (grid[index + ch] - grid[index_right + ch]);
                results[ch] += grad_val;
                idelta[ch] += grad_val * grad_val;
            }
        }

        // left side
        if (cur_d > 0) {
            pos_grid[d] = cur_d - 1;
            uint32_t index_left = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // results[ch] += w * clamp(grid[index + ch] - grid[index_left + ch], -1.0f, 1.0f);
                grad_val = (grid[index + ch] - grid[index_left + ch]);
                results[ch] += grad_val;
                idelta[ch] += grad_val * grad_val;
            }
        }

        // reset
        pos_grid[d] = cur_d;
    }

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        // index may collide, so use atomic!
        atomicAdd(&grad[index + ch], w * results[ch] * rsqrtf(idelta[ch] + 1e-9f));
    }

}


template <typename scalar_t, uint32_t D>
void kernel_grad_tv_wrapper(const scalar_t *inputs, const scalar_t *embeddings, scalar_t *grad, const int *offsets, const float weight, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grad_tv<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        case 2: kernel_grad_tv<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        case 4: kernel_grad_tv<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        case 8: kernel_grad_tv<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


template <typename scalar_t>
void grad_total_variation_cuda(const scalar_t *inputs, const scalar_t *embeddings, scalar_t *grad, const int *offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    switch (D) {
        case 2: kernel_grad_tv_wrapper<scalar_t, 2>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 3: kernel_grad_tv_wrapper<scalar_t, 3>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 4: kernel_grad_tv_wrapper<scalar_t, 4>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 5: kernel_grad_tv_wrapper<scalar_t, 5>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}


void grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad, const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "grad_total_variation", ([&] {
        grad_total_variation_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), weight, B, D, C, L, S, H, gridtype, align_corners);
    }));
}



template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_hash_reinitialize(
    const float * __restrict__ inputs, 
    scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    const uint32_t B, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp, 
    const float std,
    const bool * grid_mask
    ) {

    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    grid_mask += (uint32_t)offsets[level] * 1;

    inputs += b * D;
    // outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate (always use float for precision!)
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        // pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        // if (interp == 1) {
            // pos[d] = smoothstep(pos[d]);
        // }

    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // interpolate
    // scalar_t results[C] = {0}; // temp results in register

    // setup random seed

    // curandState state;
    // curand_init(1234+level, b, 0, &state);

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        // float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                // w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                // w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);
        uint32_t index_mask = index / 2;

        // freeze the values which is static
        if (grid_mask[index_mask]){
            // writing to register (fast)
            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // grid[index+ch] = ((curand_uniform(&state)) * 2 * std) - std;
                grid[index+ch] = std;
            }
        }
        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }    
}

template <typename scalar_t, uint32_t D>
void kernel_grid_hash_reinitialize_wrapper(const float *inputs, scalar_t *embeddings, const int *offsets, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners, const uint32_t interp, const float std, const bool* grid_mask) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid_hash_reinitialize<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, B, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        case 2: kernel_grid_hash_reinitialize<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, B, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        case 4: kernel_grid_hash_reinitialize<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, B, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        case 8: kernel_grid_hash_reinitialize<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, B, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void grid_encode_hash_reinitialize_cuda(const float *inputs, scalar_t *embeddings, const int *offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners, const uint32_t interp, const float std, const bool* grid_mask) {
    switch (D) {
        case 2: kernel_grid_hash_reinitialize_wrapper<scalar_t, 2>(inputs, embeddings, offsets, B, C, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        case 3: kernel_grid_hash_reinitialize_wrapper<scalar_t, 3>(inputs, embeddings, offsets, B, C, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        case 4: kernel_grid_hash_reinitialize_wrapper<scalar_t, 4>(inputs, embeddings, offsets, B, C, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        case 5: kernel_grid_hash_reinitialize_wrapper<scalar_t, 5>(inputs, embeddings, offsets, B, C, L, S, H, gridtype, align_corners, interp, std, grid_mask); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

void grid_encode_hash_reinitialize(const at::Tensor inputs, at::Tensor embeddings, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners, const uint32_t interp, const float std, const at::Tensor grid_mask) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grid_mask);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grid_mask);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_BOOL(grid_mask);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "grid_encode_hash_reinitialize", ([&] {
        grid_encode_hash_reinitialize_cuda<scalar_t>(inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), B, D, C, L, S, H, gridtype, align_corners, interp, std, grid_mask.data_ptr<bool>());
    }));
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_set_static(const float * __restrict__ inputs, bool * __restrict__ grid_mask, const int * __restrict__ offsets, const uint32_t B, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    grid_mask += (uint32_t)offsets[level] * 1;
    // locate
    inputs += b * D;
    // outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate (always use float for precision!)
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
    }

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        // float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                // w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                // w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, 1>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);
        // printf("index %d", index);
        grid_mask[index] = 0.0;
    }    
}

template <typename scalar_t, uint32_t D>
void kernel_grid_set_static_wrapper(const float *inputs, bool *grid_mask, const int *offsets, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid_set_static<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, grid_mask, offsets, B, L, S, H, gridtype, align_corners); break;
        case 2: kernel_grid_set_static<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, grid_mask, offsets, B, L, S, H, gridtype, align_corners); break;
        case 4: kernel_grid_set_static<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, grid_mask, offsets, B, L, S, H, gridtype, align_corners); break;
        case 8: kernel_grid_set_static<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, grid_mask, offsets, B, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void grid_encode_set_static_cuda(const float *inputs, bool *grid_mask, const int *offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    switch (D) {
        case 2: kernel_grid_set_static_wrapper<scalar_t, 2>(inputs, grid_mask, offsets, B, C, L, S, H, gridtype, align_corners); break;
        case 3: kernel_grid_set_static_wrapper<scalar_t, 3>(inputs, grid_mask, offsets, B, C, L, S, H, gridtype, align_corners); break;
        case 4: kernel_grid_set_static_wrapper<scalar_t, 4>(inputs, grid_mask, offsets, B, C, L, S, H, gridtype, align_corners); break;
        case 5: kernel_grid_set_static_wrapper<scalar_t, 5>(inputs, grid_mask, offsets, B, C, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

void grid_encode_set_static(const at::Tensor inputs, at::Tensor grid_mask, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(grid_mask);
    CHECK_CUDA(offsets);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(grid_mask);
    CHECK_CONTIGUOUS(offsets);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_BOOL(grid_mask);
    CHECK_IS_INT(offsets);
    // CHECK_IS_FLOATING(dy_dx);

    grid_encode_set_static_cuda<bool>(inputs.data_ptr<float>(), grid_mask.data_ptr<bool>(), offsets.data_ptr<int>(), B, D, C, L, S, H, gridtype, align_corners);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    // grid_mask.scalar_type(), "grid_encode_set_static", ([&] {
    //     grid_encode_set_static_cuda<scalar_t>(inputs.data_ptr<float>(), grid_mask.data_ptr<bool>(), offsets.data_ptr<int>(), B, D, C, L, S, H, gridtype, align_corners);
    // }));
}

















// grid for rectangle boxes

template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index_rect(const uint32_t gridtype, const bool align_corners, const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution[D], const uint32_t pos_grid[D])
{
    // NOTE: resolution should be a array
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += pos_grid[d] * stride;
        stride *= align_corners ? resolution[d]: (resolution[d] + 1);
    }

    if (gridtype == 0 && stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

    return (index % hashmap_size) * C + ch;
}


template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_rect(
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    scalar_t * __restrict__ dy_dx,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    // const float scale = exp2f(level * S) * H - 1.0f;
    // const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    float pos[D];
    float pos_deriv[D]; 
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos_deriv[d] = smoothstep_derivative(pos[d]);
            pos[d] = smoothstep(pos[d]);
        } else {
            pos_deriv[d] = 1.0f; // linear deriv is default to 1
        }

    }

    scalar_t results[C] = {0};

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_rect_last_nearest(
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    scalar_t * __restrict__ dy_dx,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    // const float scale = exp2f(level * S) * H - 1.0f;
    // const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    scalar_t results[C] = {0};

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << (D-1)); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];
        // pos_grid_local[D-1] = pos[D-1] < 0.5 ? pos_grid[D-1]: pos_grid[D-1] + 1;
        pos_grid_local[D-1] = pos_grid[D-1] + rintf(pos[D-1]);

        #pragma unroll
        for (uint32_t d = 0; d < D-1; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

   
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_rect_all_nearest(
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    scalar_t * __restrict__ dy_dx,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    // const float scale = exp2f(level * S) * H - 1.0f;
    // const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    float pos[D];
    float pos_deriv[D]; 
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos_deriv[d] = smoothstep_derivative(pos[d]);
            pos[d] = smoothstep(pos[d]);
        } else {
            pos_deriv[d] = 1.0f; // linear deriv is default to 1
        }

    }

    scalar_t results[C] = {0};

    uint32_t pos_grid_local[D];
    #pragma unroll
    for(uint32_t d = 0; d < D; ++d){
        pos_grid_local[d] = rintf(pos[d]) + pos_grid[d];
        // pos_grid_local[d] = pos[d] < 0.5 ? pos_grid[d]: pos_grid[d] + 1;
    }
    uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        results[ch] += 1.0 * grid[index + ch];
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

   
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward_rect(
    const scalar_t * __restrict__ grad,
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos[d] = smoothstep(pos[d]);
        }
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
            }
        }
    }
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward_rect_last_nearest(
    const scalar_t * __restrict__ grad,
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos[d] = smoothstep(pos[d]);
        }
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
    
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << (D-1)); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];
        // pos_grid_local[D-1] = pos_grid[D-1];
        // pos_grid_local[D-1] = pos[D-1] < 0.5 ? pos_grid[D-1]: pos_grid[D-1] + 1;
        pos_grid_local[D-1] = rintf(pos[D-1]) + pos_grid[D-1];

        #pragma unroll
        for (uint32_t d = 0; d < (D-1); d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
            }
        }
    }
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward_rect_all_nearest(
    const scalar_t * __restrict__ grad,
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos[d] = smoothstep(pos[d]);
        }
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
    float w = 1;
    uint32_t pos_grid_local[D];
    #pragma unroll
    for(uint32_t d = 0; d < D; ++d){
        // pos_grid_local[d] = pos[d] < 0.5 ? pos_grid[d]: pos_grid[d] + 1;
        pos_grid_local[d] = rintf(pos[d]) + pos_grid[d];
    }
    uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

    if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
    } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
            }
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grid_rect_wrapper(const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float *S, const int *H, scalar_t *dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    if(interp<2){
        switch (C) {
            case 1: kernel_grid_rect<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 2: kernel_grid_rect<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 4: kernel_grid_rect<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 8: kernel_grid_rect<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
        }
    }
    if(interp==2){//last nearest
        switch (C) {
            case 1: kernel_grid_rect_last_nearest<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 2: kernel_grid_rect_last_nearest<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 4: kernel_grid_rect_last_nearest<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 8: kernel_grid_rect_last_nearest<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
        }
    }
    if(interp==3){//all nearest
        switch (C) {
            case 1: kernel_grid_rect_all_nearest<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 2: kernel_grid_rect_all_nearest<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 4: kernel_grid_rect_all_nearest<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            case 8: kernel_grid_rect_all_nearest<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype, align_corners, interp); break;
            default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
        }
    }
}

template <typename scalar_t>
void rect_grid_encode_forward_cuda(const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *S, const int *H, scalar_t *dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    switch (D) {
        case 2: kernel_grid_rect_wrapper<scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 3: kernel_grid_rect_wrapper<scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 4: kernel_grid_rect_wrapper<scalar_t, 4>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 5: kernel_grid_rect_wrapper<scalar_t, 5>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

void rect_grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor S, const at::Tensor H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(S);
    CHECK_CUDA(H);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(S);
    CHECK_CONTIGUOUS(H);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_INT(H);
    CHECK_IS_FLOATING(S);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "rect_grid_encode_forward", ([&] {
        rect_grid_encode_forward_cuda<scalar_t>(inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, S.data_ptr<float>(), H.data_ptr<int>(), dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, gridtype, align_corners, interp);
    }));
}

template <typename scalar_t, uint32_t D>
void kernel_grid_rect_backward_wrapper(const scalar_t *grad, const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const float *S, const int *H, scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    if(interp < 2){
        switch (C) {
            case 1: 
                kernel_grid_backward_rect<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp); 
                if (dy_dx) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 2: 
                kernel_grid_backward_rect<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 4: 
                kernel_grid_backward_rect<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 8: 
                kernel_grid_backward_rect<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
        }
    }
    if(interp==2){
        switch (C) {
            case 1: 
                kernel_grid_backward_rect_last_nearest<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp); 
                if (dy_dx) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 2: 
                kernel_grid_backward_rect_last_nearest<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 4: 
                kernel_grid_backward_rect_last_nearest<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 8: 
                kernel_grid_backward_rect_last_nearest<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
        }
    }
    if(interp==3){
        switch (C) {
            case 1: 
                kernel_grid_backward_rect_all_nearest<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp); 
                if (dy_dx) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 2: 
                kernel_grid_backward_rect_all_nearest<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 4: 
                kernel_grid_backward_rect_all_nearest<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            case 8: 
                kernel_grid_backward_rect_all_nearest<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H, gridtype, align_corners, interp);
                if (dy_dx) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
                break;
            default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
        }
    }
}

template <typename scalar_t>
void rect_grid_encode_backward_cuda(const scalar_t *grad, const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *S, const int *H, scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    switch (D) {
        case 2: kernel_grid_rect_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 3: kernel_grid_rect_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 4: kernel_grid_rect_backward_wrapper<scalar_t, 4>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 5: kernel_grid_rect_backward_wrapper<scalar_t, 5>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

void rect_grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor S, const at::Tensor H, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(S);
    CHECK_CUDA(H);
    // CHECK_CUDA(dy_dx);
    // CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(S);
    CHECK_CONTIGUOUS(H);
    // CHECK_CONTIGUOUS(dy_dx);
    // CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    CHECK_IS_INT(H);
    CHECK_IS_FLOATING(S);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "rect_grid_encode_backward", ([&] {
        rect_grid_encode_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, S.data_ptr<float>(), H.data_ptr<int>(), dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, grad_inputs.has_value() ? grad_inputs.value().data_ptr<scalar_t>() : nullptr, gridtype, align_corners, interp);
    }));
    
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_rect_grad_tv(
    const scalar_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid, 
    scalar_t * __restrict__ grad, 
    const int * __restrict__ offsets, 
    const float weight,
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    const uint32_t gridtype,
    const bool align_corners
) 
{
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    if(level>5){
        return;
    }

    grid += (uint32_t)offsets[level] * C;
    grad += (uint32_t)offsets[level] * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, do nothing
    if (flag_oob) return;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    float scale[D];
    uint32_t resolution[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }
    
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D]; // [0, resolution]

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        // pos[d] -= (float)pos_grid[d]; // not used
    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // total variation on pos_grid
    scalar_t results[C] = {0}; // temp results in register
    scalar_t idelta[C] = {0};

    uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

    scalar_t w = weight / (2 * D);

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {

        uint32_t cur_d = pos_grid[d];
        scalar_t grad_val;

        // right side
        if (cur_d < resolution[d]) {
            pos_grid[d] = cur_d + 1;
            uint32_t index_right = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // results[ch] += w * clamp(grid[index + ch] - grid[index_right + ch], -1.0f, 1.0f);
                grad_val = (grid[index + ch] - grid[index_right + ch]);
                results[ch] += grad_val;
                idelta[ch] += grad_val * grad_val;
            }
        }

        // left side
        if (cur_d > 0) {
            pos_grid[d] = cur_d - 1;
            uint32_t index_left = get_grid_index_rect<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // results[ch] += w * clamp(grid[index + ch] - grid[index_left + ch], -1.0f, 1.0f);
                grad_val = (grid[index + ch] - grid[index_left + ch]);
                results[ch] += grad_val;
                idelta[ch] += grad_val * grad_val;
            }
        }

        // reset
        pos_grid[d] = cur_d;
    }

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        // index may collide, so use atomic!
        atomicAdd(&grad[index + ch], w * results[ch] * rsqrtf(idelta[ch] + 1e-9f));
    }
}

template <typename scalar_t, uint32_t D>
void kernel_rect_grad_tv_wrapper(const scalar_t *inputs, const scalar_t *embeddings, scalar_t *grad, const int *offsets, const float weight, const uint32_t B, const uint32_t C, const uint32_t L, const float *S, const int *H, const uint32_t gridtype, const bool align_corners) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_rect_grad_tv<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        case 2: kernel_rect_grad_tv<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        case 4: kernel_rect_grad_tv<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        case 8: kernel_rect_grad_tv<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, grad, offsets, weight, B, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void rect_grad_total_variation_cuda(const scalar_t *inputs, const scalar_t *embeddings, scalar_t *grad, const int *offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *S, const int *H, const uint32_t gridtype, const bool align_corners) {
    switch (D) {
        case 2: kernel_rect_grad_tv_wrapper<scalar_t, 2>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 3: kernel_rect_grad_tv_wrapper<scalar_t, 3>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 4: kernel_rect_grad_tv_wrapper<scalar_t, 4>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 5: kernel_rect_grad_tv_wrapper<scalar_t, 5>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

void rect_grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad, const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor S, const at::Tensor H, const uint32_t gridtype, const bool align_corners) {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "rect_grad_total_variation", ([&] {
        rect_grad_total_variation_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), weight, B, D, C, L, S.data_ptr<float>(), H.data_ptr<int>(), gridtype, align_corners);
    }));
}





















template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_stgrid(
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ sgrid, 
    const scalar_t * __restrict__ tgrid, 
    const scalar_t * __restrict__ mgrid, 
    const int * __restrict__ soffsets, 
    const int * __restrict__ toffsets, 
    scalar_t * __restrict__ outputs, 
    scalar_t * __restrict__ tout, 
    scalar_t * __restrict__ mout, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    const int *M,
    scalar_t * __restrict__ dy_dx,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    inputs += b * D;
    outputs += level * B * C + b * C;
    tout += level * B * C + b * C;
    mout += b * 1;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    sgrid += (uint32_t)soffsets[level] * C;
    const uint32_t shashmap_size = soffsets[level + 1] - soffsets[level];

    tgrid += (uint32_t)toffsets[level] * C;
    const uint32_t thashmap_size = toffsets[level + 1] - toffsets[level];

    float scale[D];
    uint32_t resolution[D];
    uint32_t spatial_resolution[D-1];
    uint32_t mask_resolution[D-1];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    #pragma unroll
    for (uint32_t d=0; d<D-1; ++d){
        spatial_resolution[d] = resolution[d];
    }

    #pragma unroll
    for (uint32_t d=0; d<D-1;++d)
    {
        mask_resolution[d] = (uint32_t)M[d];
    }

    float pos[D];
    uint32_t pos_grid[D];
    uint32_t pos_mask[D-1];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    #pragma unroll
    for (uint32_t d=0; d<D-1; ++d){
        pos_mask[d] = inputs[d] * (mask_resolution[d]-1) + (align_corners ? 0.0f : 0.5f);
        pos_mask[d] = rint(pos_mask[d]);
    }

    uint32_t mask_index = get_grid_index_rect<D-1, 1>(1, align_corners, 0, 2^30, mask_resolution, pos_mask);
    float mask_value = mgrid[mask_index];
    mask_value = 1./(1 + exp2f(-mask_value));
    mout[0] = mask_value;

    scalar_t results[C] = {0};
    scalar_t tresults[C] = {0};
    
    // spatial query
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << (D-1)); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D-1];

        #pragma unroll
        for (uint32_t d = 0; d < D-1; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D-1, C>(gridtype, align_corners, 0, shashmap_size, spatial_resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * sgrid[index + ch];
        }
    }

    // temporal query
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, 0, thashmap_size, resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            scalar_t temp = tgrid[index+ch];
            results[ch] += (1-mask_value) * w * temp;
            tresults[ch] += (1-mask_value) * w * temp;
        }
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        tout[ch] = tresults[ch]; 
    }
}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_stgrid_backward(
    const scalar_t * __restrict__ grad,// L x B x C
    const float * __restrict__ inputs, 
    // const scalar_t * __restrict__ sout,// L x B x C
    const scalar_t * __restrict__ tout, 
    const scalar_t * __restrict__ mout, 
    const int * __restrict__ soffsets, 
    const int * __restrict__ toffsets, 
    scalar_t * __restrict__ grad_sgrid, 
    scalar_t * __restrict__ grad_tgrid, 
    scalar_t * __restrict__ grad_mgrid, 
    const uint32_t B, const uint32_t L, const float *S, const int *H,
    const int *M,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp
)
{
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_sgrid += soffsets[level] * C;
    grad_tgrid += toffsets[level] * C;
    inputs += b * D;

    grad += level * B * C + b * C + ch; // L, B, C
    // sout += level * B * C + b * C + ch; // L, B, C
    tout += level * B * C + b * C + ch; // L, B, C
    mout += b * 1 ; // B, 1
    float mask_value = mout[0];
    // out = sout + mout * tout

    const uint32_t shashmap_size = soffsets[level + 1] - soffsets[level];
    const uint32_t thashmap_size = toffsets[level + 1] - toffsets[level];

    float scale[D];
    uint32_t resolution[D];
    uint32_t spatial_resolution[D-1];
    uint32_t mask_resolution[D-1];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) 
    {
        scale[d] = exp2f(level * S[d]) * H[d] - 1.0f; 
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    #pragma unroll
    for (uint32_t d=0; d<D-1; ++d){
        spatial_resolution[d] = resolution[d];
    }

    #pragma unroll
    for (uint32_t d=0; d<D-1;++d)
    {
        mask_resolution[d] = (uint32_t)M[d];
    }

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    float pos[D];
    uint32_t pos_grid[D];
    uint32_t pos_mask[D-1];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale[d] + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos[d] = smoothstep(pos[d]);
        }
    }

    #pragma unroll
    for (uint32_t d=0; d<D-1; ++d){
        pos_mask[d] = inputs[d] * (mask_resolution[d]-1) + (align_corners ? 0.0f : 0.5f);
        pos_mask[d] = rint(pos_mask[d]);
    }

    // spatial backward
    scalar_t grad_cur[N_C] = {0}; // fetch to register
    scalar_t tout_cur[N_C] = {0}; // fetch to register

    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        tout_cur[c] = tout[c];
    }

    // update grad_mgrid
    if(level == 0){
        uint32_t mask_index = get_grid_index_rect<D-1, 1>(1, align_corners, 0, 2^30, mask_resolution, pos_mask);
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            __half tv = 0;

            #pragma unroll
            for (uint32_t c = 0; c < N_C; ++c) {
                tv += -tout_cur[c] * grad_cur[c];
            }

            #pragma unroll
            for (uint32_t l=1; l < L; ++l){
                grad += B * C; // L, B, C
                tout += B * C; // L, B, C
                #pragma unroll
                for (uint32_t c = 0; c < N_C; ++c) {
                    tv += -grad[c] * tout[c];
                }
            }
            tv = tv * (__half(mask_value) * __half(1.-mask_value));
            atomicAdd((__half*)&grad_mgrid[mask_index], tv);
            grad -= (L-1) * B * C; // L, B, C
            tout -= (L-1) * B * C; // L, B, C
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            float tv = 0;

            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                tv += -tout_cur[c] * grad_cur[c];
            }

            #pragma unroll
            for (uint32_t l=1; l < L; ++l){
                grad += B * C; // L, B, C
                tout += B * C; // L, B, C
                #pragma unroll
                for (uint32_t c = 0; c < N_C; ++c) {
                    tv += -grad[c] * tout[c];
                }
            }
            tv = tv * mask_value * (1.-mask_value);
            atomicAdd(&grad_mgrid[mask_index], tv);
            grad -= (L-1) * B * C; // L, B, C
            tout -= (L-1) * B * C; // L, B, C
        }
    }

    // update for grad_sgrid
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << (D-1)); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D-1];

        #pragma unroll
        for (uint32_t d = 0; d < D-1; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D-1, C>(gridtype, align_corners, ch, shashmap_size, spatial_resolution, pos_grid_local);

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_sgrid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_sgrid[index + c], w * grad_cur[c]);
            }
        }
    }


    // update temporal ones
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index_rect<D, C>(gridtype, align_corners, ch, thashmap_size, resolution, pos_grid_local);

        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)((1-mask_value) * w * grad_cur[c]), (__half)((1-mask_value) * w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_tgrid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_tgrid[index + c], (1-mask_value) * w * grad_cur[c]);
            }
        }
    }
}


template <typename scalar_t, uint32_t D>
void kernel_stgrid_wrapper(
    const float *inputs, 
    const scalar_t *sembeddings, 
    const scalar_t *tembeddings, 
    const scalar_t *membeddings, 
    const int *soffsets, 
    const int *toffsets, 
    scalar_t *outputs, 
    scalar_t *tout, 
    scalar_t *mout, 
    const uint32_t B, 
    const uint32_t C, 
    const uint32_t L, 
    const float *S, 
    const int *H, 
    const int *M, 
    scalar_t *dy_dx, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_stgrid<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        case 2: kernel_stgrid<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        case 4: kernel_stgrid<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        case 8: kernel_stgrid<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void stgrid_encode_forward_cuda(
    const float *inputs, 
    const scalar_t *sembeddings, 
    const scalar_t *tembeddings, 
    const scalar_t *membeddings, 
    const int *soffsets, 
    const int *toffsets, 
    scalar_t *outputs, 
    scalar_t *tout, 
    scalar_t *mout, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t C, 
    const uint32_t L, 
    const float *S, 
    const int *H, 
    const int *M, 
    scalar_t *dy_dx, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp
) {
    switch (D) {
        case 2: kernel_stgrid_wrapper<scalar_t, 2>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, C, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        case 3: kernel_stgrid_wrapper<scalar_t, 3>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, C, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        case 4: kernel_stgrid_wrapper<scalar_t, 4>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, C, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        case 5: kernel_stgrid_wrapper<scalar_t, 5>(inputs, sembeddings, tembeddings, membeddings, soffsets, toffsets, outputs, tout, mout, B, C, L, S, H, M, dy_dx, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

void stgrid_encode_forward(
    const at::Tensor inputs, 
    const at::Tensor sembeddings, 
    const at::Tensor tembeddings, 
    const at::Tensor membeddings, 
    const at::Tensor soffsets, 
    const at::Tensor toffsets, 
    at::Tensor outputs, 
    at::Tensor tout, 
    at::Tensor mout, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t C, 
    const uint32_t L, 
    const at::Tensor S, 
    const at::Tensor H, 
    const at::Tensor M, 
    at::optional<at::Tensor> dy_dx, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp
) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(sembeddings);
    CHECK_CUDA(tembeddings);
    CHECK_CUDA(membeddings);
    CHECK_CUDA(soffsets);
    CHECK_CUDA(toffsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(tout);
    CHECK_CUDA(mout);
    CHECK_CUDA(S);
    CHECK_CUDA(H);
    CHECK_CUDA(M);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(sembeddings);
    CHECK_CONTIGUOUS(tembeddings);
    CHECK_CONTIGUOUS(membeddings);
    CHECK_CONTIGUOUS(soffsets);
    CHECK_CONTIGUOUS(toffsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(tout);
    CHECK_CONTIGUOUS(mout);
    CHECK_CONTIGUOUS(S);
    CHECK_CONTIGUOUS(H);
    CHECK_CONTIGUOUS(M);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(sembeddings);
    CHECK_IS_FLOATING(tembeddings);
    CHECK_IS_FLOATING(membeddings);
    CHECK_IS_INT(soffsets);
    CHECK_IS_INT(toffsets);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(tout);
    CHECK_IS_FLOATING(mout);
    CHECK_IS_INT(H);
    CHECK_IS_FLOATING(S);
    CHECK_IS_INT(M);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sembeddings.scalar_type(), "rect_grid_encode_forward", ([&] {
        stgrid_encode_forward_cuda<scalar_t>(
            inputs.data_ptr<float>(), 
            sembeddings.data_ptr<scalar_t>(), 
            tembeddings.data_ptr<scalar_t>(), 
            membeddings.data_ptr<scalar_t>(), 
            soffsets.data_ptr<int>(), 
            toffsets.data_ptr<int>(), 
            outputs.data_ptr<scalar_t>(), 
            tout.data_ptr<scalar_t>(), 
            mout.data_ptr<scalar_t>(), 
            B, D, C, L, S.data_ptr<float>(), H.data_ptr<int>(), M.data_ptr<int>(),
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, 
            gridtype, 
            align_corners, 
            interp
        );
    }));
}

template <typename scalar_t, uint32_t D>
void kernel_stgrid_backward_wrapper(
    const scalar_t *grad, 
    const float *inputs, 
    // const scalar_t *embeddings, 
    // const scalar_t * sout,
    const scalar_t * tout,
    const scalar_t * mout,
    const int *soffsets, 
    const int *toffsets, 
    scalar_t *grad_sembeddings, 
    scalar_t *grad_tembeddings, 
    scalar_t *grad_membeddings, 
    const uint32_t B, 
    const uint32_t C, 
    const uint32_t L, 
    const float *S, 
    const int *H, 
    const int *M,
    scalar_t *dy_dx, 
    scalar_t *grad_inputs, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp
) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 1: 
            kernel_stgrid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, L, S, H, M, gridtype, align_corners, interp); 
            if (dy_dx) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 2: 
            kernel_stgrid_backward<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, L, S, H, M, gridtype, align_corners, interp);
            if (dy_dx) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4: 
            kernel_stgrid_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, L, S, H, M, gridtype, align_corners, interp);
            if (dy_dx) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 8: 
            kernel_stgrid_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, L, S, H, M, gridtype, align_corners, interp);
            if (dy_dx) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void stgrid_encode_backward_cuda(
    const scalar_t *grad, 
    const float *inputs, 
    // const scalar_t *sout, 
    const scalar_t *tout, 
    const scalar_t *mout, 
    const int *soffsets, 
    const int *toffsets, 
    scalar_t *grad_sembeddings, 
    scalar_t *grad_tembeddings, 
    scalar_t *grad_membeddings, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t C, 
    const uint32_t L, 
    const float *S, 
    const int *H, 
    const int *M, 
    scalar_t *dy_dx, 
    scalar_t *grad_inputs, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp
) {
    switch (D) {
        case 2: kernel_stgrid_backward_wrapper<scalar_t, 2>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, C, L, S, H, M, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 3: kernel_stgrid_backward_wrapper<scalar_t, 3>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, C, L, S, H, M, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 4: kernel_stgrid_backward_wrapper<scalar_t, 4>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, C, L, S, H, M, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 5: kernel_stgrid_backward_wrapper<scalar_t, 5>(grad, inputs, tout, mout, soffsets, toffsets, grad_sembeddings, grad_tembeddings, grad_membeddings, B, C, L, S, H, M, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

void stgrid_encode_backward(
    const at::Tensor grad, 
    const at::Tensor inputs, 
    // const at::Tensor embeddings, 
    // const at::Tensor sout, 
    const at::Tensor tout, 
    const at::Tensor mout, 
    const at::Tensor soffsets, 
    const at::Tensor toffsets, 
    at::Tensor grad_sembeddings, 
    at::Tensor grad_tembeddings, 
    at::Tensor grad_membeddings, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t C, 
    const uint32_t L, 
    const at::Tensor S, 
    const at::Tensor H, 
    const at::Tensor M, 
    const at::optional<at::Tensor> dy_dx, 
    at::optional<at::Tensor> grad_inputs, 
    const uint32_t gridtype, 
    const bool align_corners, 
    const uint32_t interp
) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    // CHECK_CUDA(sout);
    CHECK_CUDA(tout);
    CHECK_CUDA(mout);
    CHECK_CUDA(soffsets);
    CHECK_CUDA(toffsets);
    CHECK_CUDA(grad_sembeddings);
    CHECK_CUDA(grad_tembeddings);
    CHECK_CUDA(grad_membeddings);
    CHECK_CUDA(S);
    CHECK_CUDA(H);
    CHECK_CUDA(M);
    // CHECK_CUDA(dy_dx);
    // CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    // CHECK_CONTIGUOUS(sout);
    CHECK_CONTIGUOUS(tout);
    CHECK_CONTIGUOUS(mout);
    CHECK_CONTIGUOUS(soffsets);
    CHECK_CONTIGUOUS(toffsets);
    CHECK_CONTIGUOUS(grad_sembeddings);
    CHECK_CONTIGUOUS(grad_tembeddings);
    CHECK_CONTIGUOUS(grad_membeddings);
    CHECK_CONTIGUOUS(S);
    CHECK_CONTIGUOUS(H);
    CHECK_CONTIGUOUS(M);
    // CHECK_CONTIGUOUS(dy_dx);
    // CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    // CHECK_IS_FLOATING(sout);
    CHECK_IS_FLOATING(tout);
    CHECK_IS_FLOATING(mout);
    CHECK_IS_INT(soffsets);
    CHECK_IS_INT(toffsets);
    CHECK_IS_FLOATING(grad_sembeddings);
    CHECK_IS_FLOATING(grad_tembeddings);
    CHECK_IS_FLOATING(grad_membeddings);
    CHECK_IS_INT(H);
    CHECK_IS_FLOATING(S);
    CHECK_IS_INT(M);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "rect_grid_encode_backward", ([&] {
        stgrid_encode_backward_cuda<scalar_t>(
            grad.data_ptr<scalar_t>(), 
            inputs.data_ptr<float>(), 
            // sout.data_ptr<scalar_t>(), 
            tout.data_ptr<scalar_t>(), 
            mout.data_ptr<scalar_t>(), 
            soffsets.data_ptr<int>(), 
            toffsets.data_ptr<int>(), 
            grad_sembeddings.data_ptr<scalar_t>(), 
            grad_tembeddings.data_ptr<scalar_t>(), 
            grad_membeddings.data_ptr<scalar_t>(), 
            B, D, C, L, S.data_ptr<float>(), H.data_ptr<int>(), M.data_ptr<int>(), 
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, 
            grad_inputs.has_value() ? grad_inputs.value().data_ptr<scalar_t>() : nullptr, 
            gridtype, 
            align_corners, 
            interp);
    }));
    
}
