/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef WASM_FP_ENVIRONMENT_H
#define WASM_FP_ENVIRONMENT_H
#include <fenv.h>
#include "wasm_exec_env.h"
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
wasm_fp_leave(const WASMFPEnvironment *env, WASMExecEnv *exec_env)
{
    bool ok = fesetenv(&env->saved) == 0;
    if (!ok && exec_env && exec_env->forge_fp_failure == WASM_FP_FAILURE_NONE)
        exec_env->forge_fp_failure = WASM_FP_FAILURE_RESTORE_FAILED;
#if defined(__x86_64__)
    _mm_setcsr(env->control);
#else
    __asm__ volatile("msr fpcr, %0" : : "r"(env->control) : "memory");
    __asm__ volatile("msr fpsr, %0" : : "r"(env->status) : "memory");
#endif
    return ok;
}

static inline bool
wasm_fp_enter(WASMFPEnvironment *env, WASMExecEnv *exec_env)
{
    if (fegetenv(&env->saved) != 0) {
        if (exec_env && exec_env->forge_fp_failure == WASM_FP_FAILURE_NONE)
            exec_env->forge_fp_failure = WASM_FP_FAILURE_ENTER_FAILED;
        return false;
    }
#if defined(__x86_64__)
    env->control = _mm_getcsr();
#else
    __asm__ volatile("mrs %0, fpcr" : "=r"(env->control));
    __asm__ volatile("mrs %0, fpsr" : "=r"(env->status));
#endif
    if (fesetenv(FE_DFL_ENV) != 0 || fesetround(FE_TONEAREST) != 0) {
        if (exec_env && exec_env->forge_fp_failure == WASM_FP_FAILURE_NONE)
            exec_env->forge_fp_failure = WASM_FP_FAILURE_ENTER_FAILED;
        wasm_fp_leave(env, exec_env);
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
