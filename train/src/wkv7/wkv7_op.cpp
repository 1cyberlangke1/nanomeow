// WKV-7 CUDA 算子的宿主包装与注册。注册成 TORCH_LIBRARY 自定义算子的目的：
// dynamo 把它当成一个不透明节点，整条递推在编译后的图里就是一个 kernel，不会断图。
#include <torch/extension.h>

struct __nv_bfloat16;
using bf = __nv_bfloat16;

void wkv7_forward_launch(int B, int T, int H, int C,
                         const float* logw, const bf* q, const bf* k, const bf* v,
                         const bf* a, const bf* b, bf* y, float* s, float* sa,
                         int chunk_len, const float* state_step);
void wkv7_backward_launch(int B, int T, int H, int C,
                          const float* logw, const bf* q, const bf* k, const bf* v,
                          const bf* a, const bf* b, const bf* dy,
                          float* s, float* sa, float* dlogw, bf* dq, bf* dk, bf* dv,
                          bf* da, bf* db, int chunk_len);

void forward(torch::Tensor &logw, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
             torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &s,
             torch::Tensor &sa, int64_t chunk_len, torch::Tensor &state_step) {
    int B = q.sizes()[0], T = q.sizes()[1], H = q.sizes()[2], C = q.sizes()[3];
    wkv7_forward_launch(B, T, H, C, (const float*)logw.data_ptr(),
                        (const bf*)q.data_ptr(), (const bf*)k.data_ptr(), (const bf*)v.data_ptr(),
                        (const bf*)a.data_ptr(), (const bf*)b.data_ptr(),
                        (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(),
                        (int)chunk_len, (const float*)state_step.data_ptr());
}

void backward(torch::Tensor &logw, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
              torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy,
              torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dlogw, torch::Tensor &dq,
              torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db,
              int64_t chunk_len) {
    int B = q.sizes()[0], T = q.sizes()[1], H = q.sizes()[2], C = q.sizes()[3];
    wkv7_backward_launch(B, T, H, C, (const float*)logw.data_ptr(),
                         (const bf*)q.data_ptr(), (const bf*)k.data_ptr(), (const bf*)v.data_ptr(),
                         (const bf*)a.data_ptr(), (const bf*)b.data_ptr(), (const bf*)dy.data_ptr(),
                         (float*)s.data_ptr(), (float*)sa.data_ptr(),
                         (float*)dlogw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(),
                         (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr(),
                         (int)chunk_len);
}

TORCH_LIBRARY(nanomeow_wkv7, m) {
    m.def("forward(Tensor logw, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, int chunk_len, Tensor state_step) -> ()");
    m.def("backward(Tensor logw, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dlogw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db, int chunk_len) -> ()");
}

TORCH_LIBRARY_IMPL(nanomeow_wkv7, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
// torch.utils.cpp_extension.load 会把这个 .pyd 当 Python 模块导入，需要一个模块入口；
// 算子本身是靠上面的 TORCH_LIBRARY 注册的，这里不用再绑定任何东西。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
