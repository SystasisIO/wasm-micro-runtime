/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef WASM_FP_ENVIRONMENT_H
#define WASM_FP_ENVIRONMENT_H
#include <fenv.h>
#if defined(__x86_64__)
#include <xmmintrin.h>
#elif !defined(__aarch64__) && !defined(__arm64__)
#error "The deterministic FP profile needs a qualified host FP environment"
#endif

typedef struct WASMFPEnvironment {
    fenv_t saved;
#if defined(__x86_64__)
    unsigned int control;
#else
    unsigned long long control;
    unsigned long long status;
#endif
} WASMFPEnvironment;

static inline bool
wasm_fp_leave(const WASMFPEnvironment *env)
{
    bool ok = fesetenv(&env->saved) == 0;
#if defined(__x86_64__)
    _mm_setcsr(env->control);
#else
    __asm__ volatile("msr fpcr, %0" : : "r"(env->control) : "memory");
    __asm__ volatile("msr fpsr, %0" : : "r"(env->status) : "memory");
#endif
    return ok;
}

static inline bool
wasm_fp_enter(WASMFPEnvironment *env)
{
    if (fegetenv(&env->saved) != 0)
        return false;
#if defined(__x86_64__)
    env->control = _mm_getcsr();
#else
    __asm__ volatile("mrs %0, fpcr" : "=r"(env->control));
    __asm__ volatile("mrs %0, fpsr" : "=r"(env->status));
#endif
    if (fesetenv(FE_DFL_ENV) != 0 || fesetround(FE_TONEAREST) != 0) {
        wasm_fp_leave(env);
        return false;
    }
    /* Disable host traps, flush-to-zero and nonstandard rounding modes. */
#if defined(__x86_64__)
    _mm_setcsr(0x1f80U);
#else
    __asm__ volatile("msr fpcr, %0" : : "r"(0ULL) : "memory");
    __asm__ volatile("msr fpsr, %0" : : "r"(0ULL) : "memory");
#endif
    return true;
}
#endif
