// WKV-7 递推的 CUDA 实现，移植自 Mini_RWKV_7 的 cuda/wkv7_cuda.cu。
// 与参考的差异只有两处，都是为了 QAT：
//   1) 衰减以 fp32 的 log_w 传入（外部已按 Q15 网格量化），kernel 里 __expf(log_w) 还原，
//      参考是从 w_in 现算 -exp(w_in)；反向里 dlogw 是「对 log_w 的梯度」（dw * w，fp32 输出），
//   2) state 的 int32 网格量化放在每个 chunk 边界（外部按 k/v 的 amax 算好 state_step 传入），
//      反向是 STE 直通，不改反向公式。
// 其余结构（每线程持 state 的一列、token 向量走 shared、chunk 末尾 dump state 给反向）
// 与参考逐行一致。
#include <cuda_bf16.h>
#include <assert.h>

using bf = __nv_bfloat16;
__device__ inline float to_float(const bf& u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float& u) { return __float2bfloat16_rn(u); }

template <int C>
__global__ void wkv7_forward_kernel(int T, int H,
                                    const float* __restrict__ logw_,
                                    const bf* __restrict__ q_, const bf* __restrict__ k_,
                                    const bf* __restrict__ v_, const bf* __restrict__ a_,
                                    const bf* __restrict__ b_, bf* __restrict__ y_,
                                    float* __restrict__ s_, float* __restrict__ sa_,
                                    int chunk_len, const float* __restrict__ state_step_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    const float state_step = state_step_[0];
    float state[C] = {0.f};
    __shared__ float qs[C], ks[C], ws[C], as[C], bs[C];

    for (int t = 0; t < T; t++) {
        int ind = ((bb * T + t) * H + hh) * C + i;
        __syncthreads();
        qs[i] = to_float(q_[ind]);
        ws[i] = __expf(logw_[ind]);
        ks[i] = to_float(k_[ind]);
        as[i] = to_float(a_[ind]);
        bs[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) sa += as[j] * state[j];
        sa_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float s = state[j];
            s = s * ws[j] + sa * bs[j] + ks[j] * v;
            state[j] = s;
            y += s * qs[j];
        }
        y_[ind] = to_bf(y);

        if ((t + 1) % chunk_len == 0) {
            if (state_step > 0.f) {
#pragma unroll
                for (int j = 0; j < C; j++) state[j] = rintf(state[j] / state_step) * state_step;
            }
            int base = ((bb * H + hh) * (T / chunk_len) + t / chunk_len) * C * C + i;
#pragma unroll
            for (int j = 0; j < C; j++) s_[base + j * C] = state[j];
        }
    }
}

template <int C>
__global__ void wkv7_backward_kernel(int T, int H,
                                     const float* __restrict__ logw_,
                                     const bf* __restrict__ q_, const bf* __restrict__ k_,
                                     const bf* __restrict__ v_, const bf* __restrict__ a_,
                                     const bf* __restrict__ b_, const bf* __restrict__ dy_,
                                     float* __restrict__ s_, float* __restrict__ sa_,
                                     float* __restrict__ dlogw_, bf* __restrict__ dq_,
                                     bf* __restrict__ dk_, bf* __restrict__ dv_,
                                     bf* __restrict__ da_, bf* __restrict__ db_,
                                     int chunk_len) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0.f}, dstate[C] = {0.f}, dstateT[C] = {0.f};
    __shared__ float ws[C], qs[C], ks[C], vs[C], as[C], bs[C], dys[C], sas[C], dSbs[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T - 1; t >= 0; t--) {
        int ind = ((bb * T + t) * H + hh) * C + i;
        __syncthreads();
        qs[i] = qi = to_float(q_[ind]);
        float wi_fac = logw_[ind];
        ws[i] = wi = __expf(wi_fac);
        ks[i] = ki = to_float(k_[ind]);
        as[i] = ai = to_float(a_[ind]);
        bs[i] = bi = to_float(b_[ind]);
        vs[i] = to_float(v_[ind]);
        dys[i] = dyi = to_float(dy_[ind]);
        sas[i] = sa_[ind];
        __syncthreads();

        if ((t + 1) % chunk_len == 0) {
            int base = ((bb * H + hh) * (T / chunk_len) + t / chunk_len) * C * C + i * C;
#pragma unroll
            for (int j = 0; j < C; j++) stateT[j] = s_[base + j];
        }

        float dq = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) dq += stateT[j] * dys[j];
        dq_[ind] = to_bf(dq);

        float iwi = 1.f / wi;
#pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki * vs[j] - bi * sas[j]) * iwi;
            dstate[j] += dyi * qs[j];
            dstateT[j] += qi * dys[j];
        }

        float dw = 0.f, dk = 0.f, dv = 0.f, db = 0.f, dSb = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * vs[j];
            dv += dstate[j] * ks[j];
            dSb += dstate[j] * bs[j];
            db += dstateT[j] * sas[j];
        }
        dlogw_[ind] = dw * wi;
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSbs[i] = dSb;
        __syncthreads();

        float da = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) da += stateT[j] * dSbs[j];
        da_[ind] = to_bf(da);

#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j] * ws[j] + dSb * as[j];
            dstateT[j] = dstateT[j] * wi + ai * dSbs[j];
        }
    }
}

void wkv7_forward_launch(int B, int T, int H, int C,
                         const float* logw, const bf* q, const bf* k, const bf* v,
                         const bf* a, const bf* b, bf* y, float* s, float* sa,
                         int chunk_len, const float* state_step) {
    assert(C == 8);
    dim3 grid(H, B), block(C);
    wkv7_forward_kernel<8><<<grid, block>>>(T, H, logw, q, k, v, a, b, y, s, sa, chunk_len, state_step);
}

void wkv7_backward_launch(int B, int T, int H, int C,
                          const float* logw, const bf* q, const bf* k, const bf* v,
                          const bf* a, const bf* b, const bf* dy,
                          float* s, float* sa, float* dlogw, bf* dq, bf* dk, bf* dv,
                          bf* da, bf* db, int chunk_len) {
    assert(C == 8);
    dim3 grid(H, B), block(C);
    wkv7_backward_kernel<8><<<grid, block>>>(T, H, logw, q, k, v, a, b, dy, s, sa,
                                             dlogw, dq, dk, dv, da, db, chunk_len);
}