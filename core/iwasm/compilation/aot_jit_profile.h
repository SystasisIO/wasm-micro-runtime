#ifndef WAMR_AOT_JIT_PROFILE_H
#define WAMR_AOT_JIT_PROFILE_H

#include "aot_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Immutable, half-open native intervals. Offsets refer to the original WASM
   file, not the Code section. Unknown PCs translate to UINT32_MAX. */
typedef struct AOTJitProfileInterval {
    uintptr_t begin;
    uintptr_t end;
    uint32 wasm_offset;
} AOTJitProfileInterval;

/* Compiler-private opt-in, before aot_compile_wasm, on a caller-thread JIT.
   The loader owns the context; the map ends at aot_destroy_comp_context. */
bool aot_jit_profile_enable(AOTCompContext *comp_ctx);

bool aot_jit_profile_location(AOTCompContext *comp_ctx, AOTFuncContext *func_ctx,
                              const uint8 *opcode);

bool aot_jit_profile_finalize(AOTCompContext *comp_ctx);

/* Release unfinished metadata while its LLVM module/context are still alive.
   The profile/plugin itself must outlive ORC resource removal. */
void aot_jit_profile_discard_metadata(AOTCompContext *comp_ctx);

/* The loader publishes only after all functions/stack metadata and its final
   owner check succeed. Emission alone does not make the map observable. */
bool aot_jit_profile_publish(AOTCompContext *comp_ctx);

void aot_jit_profile_destroy(AOTCompContext *comp_ctx);

/* Allocation-free immutable views, visible only after the loader publishes a
   complete compilation. A NULL context is an empty map (including modules with
   no defined functions). Callers must pin the owner and exclude compile/unload
   while reading; no observer prolongs the context lifetime on its own. */
const AOTJitProfileInterval *aot_jit_profile_intervals(const AOTCompContext *comp_ctx,
                                                       uint32 *count);

uint32 aot_jit_profile_translate(const AOTCompContext *comp_ctx, uintptr_t pc);

#ifdef __cplusplus
}
#endif
#endif
