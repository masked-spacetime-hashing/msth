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
    mask_value = 1./(1 + __expf(-mask_value));
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