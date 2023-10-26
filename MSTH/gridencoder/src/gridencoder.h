#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp);
void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp);

void grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad, const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners);

void grid_encode_hash_reinitialize(const at::Tensor inputs, at::Tensor embeddings, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners, const uint32_t interp, const float std, const at::Tensor grid_mask);

void grid_encode_set_static(const at::Tensor inputs, at::Tensor grid_mask, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners);



// rectangle grid

void rect_grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor S, const at::Tensor H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp);

void rect_grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor S, const at::Tensor H, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp);

void rect_grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad, const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor S, const at::Tensor H, const uint32_t gridtype, const bool align_corners);

// stgrid
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
);

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
);

#endif