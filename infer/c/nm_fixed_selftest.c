/* 定点底座自测：从 stdin 读一行一条指令，算完把结果打到 stdout。
 * 指令与 infer/ref/fixed.py 一一对应，供 infer/tests/test_c_fixed.py 与 Python 侧对拍。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nm_fixed.h"

int nm_range_error = 0;

int main(void)
{
    char op[16];
    while (scanf("%15s", op) == 1) {
        if (strcmp(op, "rd") == 0) {                 /* round_div(num, den) */
            long long num, den;
            if (scanf("%lld %lld", &num, &den) != 2) return 2;
            printf("%lld\n", (long long)nm_round_div(num, den));
        } else if (strcmp(op, "nm") == 0) {          /* mul((m1,e1), (m2,e2)) */
            long long m1, e1, m2, e2;
            if (scanf("%lld %lld %lld %lld", &m1, &e1, &m2, &e2) != 4) return 2;
            nm_scale a = {(int32_t)m1, (int32_t)e1}, b = {(int32_t)m2, (int32_t)e2};
            nm_scale r = nm_mul(a, b);
            printf("%d %d\n", r.m, r.e);
        } else if (strcmp(op, "nd") == 0) {          /* div((m1,e1), (m2,e2)) */
            long long m1, e1, m2, e2;
            if (scanf("%lld %lld %lld %lld", &m1, &e1, &m2, &e2) != 4) return 2;
            nm_scale a = {(int32_t)m1, (int32_t)e1}, b = {(int32_t)m2, (int32_t)e2};
            nm_scale r = nm_div(a, b);
            printf("%d %d\n", r.m, r.e);
        } else if (strcmp(op, "nas") == 0) {         /* apply_scale(val, (m,e)) */
            long long val, m, e;
            if (scanf("%lld %lld %lld", &val, &m, &e) != 3) return 2;
            nm_scale s = {(int32_t)m, (int32_t)e};
            printf("%lld\n", (long long)nm_apply_scale(val, s));
        } else if (strcmp(op, "nmi") == 0) {         /* mul_int((m,e), k) */
            long long m, e, k;
            if (scanf("%lld %lld %lld", &m, &e, &k) != 3) return 2;
            nm_scale s = {(int32_t)m, (int32_t)e};
            nm_scale r = nm_mul_int(s, k);
            printf("%d %d\n", r.m, r.e);
        } else if (strcmp(op, "ndi") == 0) {         /* div_int((m,e), k) */
            long long m, e, k;
            if (scanf("%lld %lld %lld", &m, &e, &k) != 3) return 2;
            nm_scale s = {(int32_t)m, (int32_t)e};
            nm_scale r = nm_div_int(s, k);
            printf("%d %d\n", r.m, r.e);
        } else if (strcmp(op, "nrs") == 0) {         /* rescale(val, (m,e), frac) */
            long long val, m, e, frac;
            if (scanf("%lld %lld %lld %lld", &val, &m, &e, &frac) != 4) return 2;
            nm_scale s = {(int32_t)m, (int32_t)e};
            printf("%lld\n", (long long)nm_rescale(val, s, (int)frac));
        } else if (strcmp(op, "nrc") == 0) {         /* requant_code(v, num) */
            long long v, num;
            if (scanf("%lld %lld", &v, &num) != 2) return 2;
            printf("%lld\n", (long long)nm_requant_code(v, num));
        } else if (strcmp(op, "nsf") == 0) {         /* scale_from_frac(f) */
            long long f;
            if (scanf("%lld", &f) != 1) return 2;
            nm_scale r = nm_scale_from_frac((int)f);
            printf("%d %d\n", r.m, r.e);
        } else {
            fprintf(stderr, "未知指令 %s\n", op);
            return 2;
        }
    }
    return nm_range_error ? 1 : 0;
}