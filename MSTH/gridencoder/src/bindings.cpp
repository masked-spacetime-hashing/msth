#include <torch/extension.h>

#include "gridencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
    m.def("grad_total_variation", &grad_total_variation, "grad_total_variation (CUDA)");
    m.def("grid_encode_hash_reinitialize", &grid_encode_hash_reinitialize, "grid_encode_hash_reinitialize (CUDA)");
    m.def("grid_encode_set_static", &grid_encode_set_static, "grid_encode_set_static (CUDA)");
    m.def("rect_grid_encode_forward", &rect_grid_encode_forward, "rect_grid_encode_forward (CUDA)");
    m.def("rect_grid_encode_backward", &rect_grid_encode_backward, "rect_grid_encode_backward (CUDA)");
    m.def("rect_grad_total_variation", &rect_grad_total_variation, "rect_grad_total_variation (CUDA)");
    m.def("stgrid_encode_forward", &stgrid_encode_forward, "stgrid_encode_forward (CUDA)");
    m.def("stgrid_encode_backward", &stgrid_encode_backward, "stgrid_encode_backward (CUDA)");
}