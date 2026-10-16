#ifndef WAMR_AOT_LLVM_EXTRA_H
#define WAMR_AOT_LLVM_EXTRA_H

#include "llvm-c/LLJIT.h"
#include "wasm_export.h"

LLVM_C_EXTERN_C_BEGIN

/* Configure an eager, caller-thread LLJIT before construction. The callback
   context must remain valid until synchronous JIT destruction completes. */
LLVMErrorRef
aot_configure_bounded_jit_builder(LLVMOrcLLJITBuilderRef builder,
                                 const WASMLoadBudget *budget,
                                 uint64_t native_map_limit);

LLVM_C_EXTERN_C_END
#endif
