#ifndef WAMR_AOT_LLVM_EXTRA_H
#define WAMR_AOT_LLVM_EXTRA_H

#include "llvm-c/LLJIT.h"
#include "wasm_export.h"
#include "../interpreter/wasm.h"

LLVM_C_EXTERN_C_BEGIN

/* Configure an eager, caller-thread LLJIT before construction. The callback
   context must remain valid until synchronous JIT destruction completes. */
LLVMErrorRef
aot_configure_bounded_jit_builder(LLVMOrcLLJITBuilderRef builder,
                                 const WASMLoadBudget *budget,
                                 uint64_t native_map_limit);

/* Only for the caller-thread LLJIT built by the bounded constructor above.
   The owner must remain pinned; no allocation/finalization may race this read.
   NONE here describes maps only; module publication must also check bodies. */
WASMNativeCodePolicy
aot_get_bounded_jit_native_code_policy(LLVMOrcLLJITRef jit);

LLVM_C_EXTERN_C_END
#endif
