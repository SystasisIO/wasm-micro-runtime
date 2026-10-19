/*
 * Copyright (C) 2019 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/Error.h>
#if LLVM_VERSION_MAJOR < 17
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/Triple.h>
#endif
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#if LLVM_VERSION_MAJOR < 17
#include <llvm-c/Initialization.h>
#endif
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#if LLVM_VERSION_MAJOR >= 17
#include <llvm/Support/PGOOptions.h>
#include <llvm/Support/VirtualFileSystem.h>
#endif
#include <llvm/Target/CodeGenCWrappers.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#if LLVM_VERSION_MAJOR >= 17
#include <llvm/TargetParser/Triple.h>
#endif
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>
#include <llvm/Transforms/Vectorize/LoopVectorize.h>
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>
#include <llvm/Transforms/Vectorize/VectorCombine.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#if LLVM_VERSION_MAJOR >= 12
#include <llvm/Analysis/AliasAnalysis.h>
#endif
#include <llvm/ProfileData/InstrProf.h>

#include <cstring>
#include "../aot/aot_runtime.h"
#include "aot_llvm.h"
#include "aot_llvm_extra.h"

#if LLVM_VERSION_MAJOR >= 22
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/TaskDispatch.h>
#include <array>
#include <limits>
#include <new>
#endif

using namespace llvm;
using namespace llvm::orc;

#if LLVM_VERSION_MAJOR >= 22 \
    && (defined(BH_PLATFORM_DARWIN) || defined(BH_PLATFORM_LINUX))
namespace {

Error jit_memory_error(const char *text)
{
    return make_error<StringError>(text, inconvertibleErrorCode());
}

/* Only bridge-owned records use this allocator. LLVM's own IR, optimizer and
   object-buffer heap is deliberately outside the native preparation ceiling. */
struct JitOwned {
    static void *operator new(size_t size)
    {
        if (size > UINT32_MAX)
            throw std::bad_alloc();
        void *p = wasm_runtime_malloc((uint32)size);
        if (!p)
            throw std::bad_alloc();
        return p;
    }
    static void operator delete(void *p) { wasm_runtime_free(p); }
};

class BoundedJitMemory final : public jitlink::JITLinkMemoryManager,
                             public JitOwned {
    struct Segment {
        void *address = nullptr;
        char *working = nullptr;
        size_t size = 0;
        void *token = nullptr;
        int flags = 0;
        int initial_protection = 0;
        int protection = 0;
        bool temporary = false;
        bool jit = false;
    };
    struct Action {
        Action *next;
        ExecutorAddr callee;
        size_t size;
        /* Serialized LLVM wrapper arguments immediately follow this record. */
    };
    struct Record : JitOwned {
        Record *next = nullptr;
        std::array<Segment, AllocGroup::NumGroups> segments{};
        size_t count = 0;
        Action *actions = nullptr;
        bool registered = false;
        bool committed = false;
    };
    class Flight final : public InFlightAlloc, public JitOwned {
      public:
        Flight(BoundedJitMemory &owner, Record *record, jitlink::LinkGraph &graph)
          : owner(owner), record(record), graph(graph) {}
        ~Flight() override
        {
            if (record)
                consumeError(owner.destroy(record));
        }
        void abandon(OnAbandonedFunction done) override
        {
            auto *p = record;
            record = nullptr;
            done(p ? owner.destroy(p) : Error::success());
        }
        void finalize(OnFinalizedFunction done) override
        {
            if (auto error = owner.finalize(*record, graph)) {
                auto *p = record;
                record = nullptr;
                done(joinErrors(std::move(error), owner.destroy(p)));
                return;
            }
            auto *p = record;
            record = nullptr;
            done(FinalizedAlloc(ExecutorAddr::fromPtr(p)));
        }
      private:
        BoundedJitMemory &owner;
        Record *record;
        jitlink::LinkGraph &graph;
    };

  public:
    BoundedJitMemory(WASMLoadBudget budget, uint64_t limit)
      : budget(budget), limit(limit) {}
    ~BoundedJitMemory() override
    {
        /* No deferred LLVM worker may outlive this manager or callback context.
           Failed physical unmaps stay charged in the existing load owner. */
        while (records)
            consumeError(destroy(records));
    }
    void allocate(const jitlink::JITLinkDylib *, jitlink::LinkGraph &graph,
                  OnAllocatedFunction done) override
    {
        done(allocate(graph));
    }
    void deallocate(std::vector<FinalizedAlloc> allocations,
                    OnDeallocatedFunction done) override
    {
        Error error = Error::success();
        for (auto &allocation : llvm::reverse(allocations))
            error = joinErrors(std::move(error), destroy(allocation.release().toPtr<Record *>()));
        done(std::move(error));
    }

    WASMNativeCodePolicy native_code_policy() const
    {
        auto policy = WASM_NATIVE_CODE_NONE;
        for (const auto *record = records; record; record = record->next) {
            if (!record->committed)
                return WASM_NATIVE_CODE_UNCOMMITTED;
            for (size_t i = 0; i < record->count; ++i) {
                const auto &segment = record->segments[i];
                if (segment.temporary) {
                    if (segment.address)
                        return WASM_NATIVE_CODE_UNCOMMITTED;
                    continue;
                }
                if (!segment.address)
                    return WASM_NATIVE_CODE_UNCOMMITTED;
                if (!(segment.protection & MMAP_PROT_EXEC))
                    continue;
                if (segment.protection != (MMAP_PROT_READ | MMAP_PROT_EXEC))
                    return WASM_NATIVE_CODE_UNCOMMITTED;
                if (segment.jit) {
#if defined(BH_PLATFORM_DARWIN) && defined(BUILD_TARGET_AARCH64)
                    if ((segment.flags & MMAP_MAP_NO_JIT)
                        || segment.initial_protection
                               != (MMAP_PROT_READ | MMAP_PROT_WRITE
                                   | MMAP_PROT_EXEC))
                        return WASM_NATIVE_CODE_UNCOMMITTED;
#else
                    return WASM_NATIVE_CODE_UNCOMMITTED;
#endif
                }
                else if (!(segment.flags & MMAP_MAP_NO_JIT)
                         || segment.initial_protection
                                != (MMAP_PROT_READ | MMAP_PROT_WRITE))
                    return WASM_NATIVE_CODE_UNCOMMITTED;
                const auto current = segment.jit ? WASM_NATIVE_CODE_MAP_JIT
                                                 : WASM_NATIVE_CODE_RX;
                if (policy != WASM_NATIVE_CODE_NONE && policy != current)
                    return WASM_NATIVE_CODE_UNCOMMITTED;
                policy = current;
            }
        }
        return policy;
    }

  private:
    AllocResult allocate(jitlink::LinkGraph &graph)
    {
        Record *record = nullptr;
        try {
            if (!budget.check(budget.context))
                return jit_memory_error("JIT preparation canceled or unavailable");
            record = new Record();
            record->next = records;
            records = record;
            jitlink::BasicLayout layout(graph);
            uint64_t pending_peak = 0;
            for (auto &entry : layout.segments()) {
                auto &source = entry.second;
                if (source.Alignment > os_getpagesize()
                    || source.ZeroFillSize > SIZE_MAX - source.ContentSize)
                    return failed(record, jit_memory_error("JIT segment extent is invalid"));
                const size_t bytes = source.ContentSize + source.ZeroFillSize;
                if (!bytes)
                    continue;
                if (record->count == record->segments.size())
                    return failed(record, jit_memory_error("Too many LLVM allocation groups"));
                auto &segment = record->segments[record->count++];
                auto protection = entry.first.getMemProt();
                segment.protection =
                    ((protection & MemProt::Read) != MemProt::None ? MMAP_PROT_READ : 0)
                    | ((protection & MemProt::Write) != MemProt::None ? MMAP_PROT_WRITE : 0)
                    | ((protection & MemProt::Exec) != MemProt::None ? MMAP_PROT_EXEC : 0);
                segment.temporary = entry.first.getMemLifetime() == MemLifetime::Finalize;
                if ((segment.protection & MMAP_PROT_WRITE)
                    && (segment.protection & MMAP_PROT_EXEC))
                    return failed(record, jit_memory_error("Writable executable JIT segment refused"));
                if (auto error = reserve(segment, bytes, pending_peak))
                    return failed(record, std::move(error));
            }
            /* All segment extents, working buffers and map tokens have passed
               the same owner before the first actual mapping. */
            size_t index = 0;
            for (auto &entry : layout.segments()) {
                auto &source = entry.second;
                if (!source.ContentSize && !source.ZeroFillSize)
                    continue;
                auto &segment = record->segments[index++];
                if (auto error = map(segment))
                    return failed(record, std::move(error));
                source.Addr = ExecutorAddr::fromPtr(segment.address);
                source.WorkingMem = segment.working;
            }
            if (auto error = layout.apply())
                return failed(record, std::move(error));
            auto flight = std::unique_ptr<InFlightAlloc>(new Flight(*this, record, graph));
            record = nullptr;
            return flight;
        }
        catch (const std::bad_alloc &) {
            budget.allocation_failed(budget.context);
            return failed(record, jit_memory_error("JIT preparation allocation failed"));
        }
    }
    AllocResult failed(Record *record, Error error)
    {
        if (record)
            error = joinErrors(std::move(error), destroy(record));
        return std::move(error);
    }
    Error reserve(Segment &segment, size_t bytes, uint64_t &pending_peak)
    {
        const bool executable = (segment.protection & MMAP_PROT_EXEC) != 0;
        int flags = MMAP_MAP_NO_HUGE | MMAP_MAP_NO_JIT;
        int protection = MMAP_PROT_READ | MMAP_PROT_WRITE;
#if defined(BH_PLATFORM_DARWIN) && defined(BUILD_TARGET_AARCH64)
        if (executable && os_thread_jit_write_protect_supported()) {
            flags = MMAP_MAP_NO_HUGE;
            protection |= MMAP_PROT_EXEC;
            segment.jit = true;
        }
#endif
        size_t retained, peak;
        if (!os_mmap_sizes(bytes, MMAP_MAP_NO_HUGE, &retained, &peak)
            || peak > limit || pending_peak > limit - peak
            || mapped > limit - pending_peak - peak) {
            budget.allocation_failed(budget.context);
            return jit_memory_error("JIT native mapping ceiling exceeded");
        }
        /* Code fixups use a separate NX working buffer. Its actual WAMR
           allocation overlaps the native map and is charged before mapping. */
        if (executable) {
            if (retained > UINT32_MAX)
                return jit_memory_error("JIT working extent is invalid");
            segment.working = static_cast<char *>(wasm_runtime_malloc((uint32)retained));
            if (!segment.working)
                return jit_memory_error("JIT working allocation failed");
            memset(segment.working, 0, retained);
        }
        void *token = budget.begin_mapping(budget.context, retained, peak);
        if (!token)
            return jit_memory_error("JIT mapping reservation refused");
        segment.token = token;
        segment.size = retained;
        segment.flags = flags;
        segment.initial_protection = protection;
        pending_peak += peak;
        return Error::success();
    }
    Error map(Segment &segment)
    {
        if (!budget.check(budget.context))
            return jit_memory_error("JIT preparation canceled or unavailable");
        void *address = os_mmap(nullptr, segment.size, segment.initial_protection,
                                segment.flags, os_get_invalid_handle());
        budget.complete_mapping(budget.context, segment.token, address);
        segment.token = nullptr;
        if (!address) {
            budget.allocation_failed(budget.context);
            return jit_memory_error("JIT native mapping failed");
        }
        segment.address = address;
        mapped += segment.size;
        if (!(segment.protection & MMAP_PROT_EXEC))
            segment.working = static_cast<char *>(address);
        return Error::success();
    }
    Error finalize(Record &record, jitlink::LinkGraph &graph)
    {
        if (!budget.check(budget.context))
            return jit_memory_error("JIT preparation canceled or unavailable");
        /* Preallocate retained unwind action records before registration. */
        for (auto &pair : graph.allocActions()) {
            if (!pair.Dealloc)
                continue;
            const auto &arguments = pair.Dealloc.getArgData();
            if (arguments.size() > UINT32_MAX - sizeof(Action))
                return jit_memory_error("JIT unwind action extent is invalid");
            auto *action = static_cast<Action *>(wasm_runtime_malloc((uint32)(sizeof(Action) + arguments.size())));
            if (!action)
                return jit_memory_error("JIT unwind action allocation failed");
            action->callee = pair.Dealloc.getCallee();
            action->size = arguments.size();
            memcpy(action + 1, arguments.data(), action->size);
            action->next = record.actions;
            record.actions = action;
        }
        for (size_t i = 0; i < record.count; ++i) {
            auto &segment = record.segments[i];
            if (!budget.before_protect(budget.context))
                return jit_memory_error("JIT segment protection refused");
            const bool code = segment.protection & MMAP_PROT_EXEC;
            if (code) {
#if defined(BH_PLATFORM_DARWIN) && defined(BUILD_TARGET_AARCH64)
                if (segment.jit && !os_thread_jit_write_protect_supported()) {
                    budget.protection_failed(budget.context);
                    return jit_memory_error("Native MAP_JIT protection is unavailable");
                }
                const bool previous = segment.jit
                                          ? os_thread_jit_write_protect_state()
                                          : true;
                if (segment.jit)
                    os_thread_jit_write_protect_np(false);
#endif
                memcpy(segment.address, segment.working, segment.size);
                sys::Memory::InvalidateInstructionCache(segment.address, segment.size);
#if defined(BH_PLATFORM_DARWIN) && defined(BUILD_TARGET_AARCH64)
                if (segment.jit)
                    os_thread_jit_write_protect_np(previous);
#endif
            }
            if (!segment.jit && os_mprotect(segment.address, segment.size, segment.protection) != 0) {
                budget.protection_failed(budget.context);
                return jit_memory_error("JIT native protection failed");
            }
            if (code) {
                wasm_runtime_free(segment.working);
                segment.working = nullptr;
            }
        }
        auto actions = shared::runFinalizeActions(graph.allocActions());
        if (!actions) {
            consumeError(discard_actions(record, false));
            return actions.takeError();
        }
        /* From here every saved action is registered and must be undone once. */
        record.registered = true;
        if (!budget.check(budget.context))
            return jit_memory_error("JIT preparation canceled or unavailable");
        for (size_t i = 0; i < record.count; ++i)
            if (record.segments[i].temporary)
                if (auto error = unmap(record.segments[i]))
                    return error;
        if (!budget.check(budget.context))
            return jit_memory_error("JIT preparation canceled or unavailable");
        record.committed = true;
        return Error::success();
    }
    Error discard_actions(Record &record, bool run)
    {
        Error error = Error::success();
        while (auto *action = record.actions) {
            record.actions = action->next;
            if (run) {
                shared::WrapperFunctionCall::ArgDataBufferType arguments;
                const char *bytes = reinterpret_cast<const char *>(action + 1);
                arguments.append(bytes, bytes + action->size);
                shared::WrapperFunctionCall call(action->callee, std::move(arguments));
                error = joinErrors(std::move(error), call.runWithSPSRetErrorMerged());
            }
            wasm_runtime_free(action);
        }
        return error;
    }
    Error unmap(Segment &segment)
    {
        if (segment.token) {
            budget.complete_mapping(budget.context, segment.token, nullptr);
            segment.token = nullptr;
        }
        if (segment.working && segment.working != segment.address)
            wasm_runtime_free(segment.working);
        segment.working = nullptr;
        if (!segment.address)
            return Error::success();
        const bool released = budget.unmap(budget.context, segment.address, segment.size);
        segment.address = nullptr;
        if (!released)
            return jit_memory_error("JIT physical unmapping requires owner cleanup retry");
        mapped -= segment.size;
        return Error::success();
    }
    Error destroy(Record *record)
    {
        Error error = discard_actions(*record, record->registered);
        for (size_t i = record->count; i != 0; --i)
            error = joinErrors(std::move(error), unmap(record->segments[i - 1]));
        Record **slot = &records;
        while (*slot != record)
            slot = &(*slot)->next;
        *slot = record->next;
        delete record;
        return error;
    }
    WASMLoadBudget budget;
    uint64_t limit;
    uint64_t mapped = 0;
    Record *records = nullptr;
};

} // namespace
#endif

LLVMErrorRef
aot_configure_bounded_jit_builder(LLVMOrcLLJITBuilderRef builder,
                                 const WASMLoadBudget *budget,
                                 uint64_t native_map_limit)
{
#if LLVM_VERSION_MAJOR >= 22 \
    && (defined(BH_PLATFORM_DARWIN) || defined(BH_PLATFORM_LINUX))
    if (!builder || !budget || !budget->begin_mapping || !budget->complete_mapping
        || !budget->unmap || !budget->check || !budget->before_protect
        || !budget->allocation_failed || !budget->protection_failed
        || !native_map_limit || native_map_limit > (uint64_t{1} << 30))
        return wrap(jit_memory_error("Invalid bounded JIT configuration"));
    try {
        auto &config = *reinterpret_cast<LLJITBuilder *>(builder);
        if (!config.JTMB) {
            auto target = JITTargetMachineBuilder::detectHost();
            if (!target)
                return wrap(target.takeError());
            config.setJITTargetMachineBuilder(std::move(*target));
        }
        /* Independent code/data VMAs have no address-proximity guarantee.
           Large/PIC codegen avoids introducing a hidden arena-layout limit. */
        config.JTMB->setCodeModel(CodeModel::Large);
        config.JTMB->setRelocationModel(Reloc::PIC_);
        config.JTMB->setCodeGenOptLevel(CodeGenOptLevel::Aggressive);
#if defined(BH_PLATFORM_DARWIN)
        /* Independent native mappings need full-width DWARF EH references,
           not Mach-O compact-unwind's 32-bit image-relative offsets. */
        config.JTMB->getOptions().MCOptions.EmitDwarfUnwind =
            EmitDwarfUnwindType::Always;
        config.setPlatformSetUp([](LLJIT &jit) -> Expected<JITDylibSP> {
            auto frames = EHFrameRegistrationPlugin::Create(
                jit.getExecutionSession());
            if (!frames)
                return frames.takeError();
            cast<ObjectLinkingLayer>(jit.getObjLinkingLayer())
                .addPlugin(std::move(*frames));
            /* WAMR owns WASM start/initialization. There are no LLVM global
               constructors to scrape; process-symbol lookup is independent. */
            return setUpInactivePlatform(jit);
        });
#endif
        auto manager = std::unique_ptr<BoundedJitMemory>(new BoundedJitMemory(*budget, native_map_limit));
        auto control = SelfExecutorProcessControl::Create(nullptr,
            std::make_unique<InPlaceTaskDispatcher>(), std::move(manager));
        if (!control)
            return wrap(control.takeError());
        config.setExecutorProcessControl(std::move(*control));
        config.setNumCompileThreads(0);
        config.setSupportConcurrentCompilation(false);
        config.setObjectLinkingLayerCreator([](ExecutionSession &session)
            -> Expected<std::unique_ptr<ObjectLayer>> {
            return std::make_unique<ObjectLinkingLayer>(session);
        });
        return LLVMErrorSuccess;
    }
    catch (const std::bad_alloc &) {
        budget->allocation_failed(budget->context);
        return wrap(jit_memory_error("JIT builder allocation failed"));
    }
#else
    (void)builder;
    (void)budget;
    (void)native_map_limit;
    return LLVMCreateStringError("Bounded LLVM JIT requires LLVM 22 on a supported POSIX target");
#endif
}

WASMNativeCodePolicy
aot_get_bounded_jit_native_code_policy(LLVMOrcLLJITRef jit)
{
#if LLVM_VERSION_MAJOR >= 22 \
    && (defined(BH_PLATFORM_DARWIN) || defined(BH_PLATFORM_LINUX))
    if (!jit)
        return WASM_NATIVE_CODE_UNCOMMITTED;
    auto &control = reinterpret_cast<LLJIT *>(jit)
                        ->getExecutionSession().getExecutorProcessControl();
    /* The bounded constructor installed this exact manager, not an arbitrary
       LLVM allocator. The WAMR handle checks that provenance before calling. */
    return static_cast<const BoundedJitMemory &>(control.getMemMgr())
        .native_code_policy();
#else
    (void)jit;
    return WASM_NATIVE_CODE_UNCOMMITTED;
#endif
}

#if LLVM_VERSION_MAJOR >= 17
namespace llvm {
template<typename T>
using Optional = std::optional<T>;
}
#endif

LLVM_C_EXTERN_C_BEGIN

bool
aot_check_simd_compatibility(LLVMTargetMachineRef target_machine);

bool
aot_apply_llvm_new_pass_manager(AOTCompContext *comp_ctx, LLVMModuleRef module);

LLVM_C_EXTERN_C_END

class ExpandMemoryOpPass
#if LLVM_VERSION_MAJOR >= 23
  : public OptionalPassInfoMixin<ExpandMemoryOpPass>
#else
  : public PassInfoMixin<ExpandMemoryOpPass>
#endif
{
  public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

PreservedAnalyses
ExpandMemoryOpPass::run(Function &F, FunctionAnalysisManager &AM)
{
    SmallVector<MemIntrinsic *, 16> MemCalls;

    /* Iterate over all instructions in the function, looking for memcpy,
     * memmove, and memset.  When we find one, expand it into a loop. */

    for (auto &BB : F) {
        for (auto &Inst : BB) {
            if (auto *Memcpy = dyn_cast_or_null<MemCpyInst>(&Inst)) {
                MemCalls.push_back(Memcpy);
            }
            else if (auto *Memmove = dyn_cast_or_null<MemMoveInst>(&Inst)) {
                MemCalls.push_back(Memmove);
            }
            else if (auto *Memset = dyn_cast_or_null<MemSetInst>(&Inst)) {
                MemCalls.push_back(Memset);
            }
        }
    }

    for (MemIntrinsic *MemCall : MemCalls) {
        if (MemCpyInst *Memcpy = dyn_cast<MemCpyInst>(MemCall)) {
            Function *ParentFunc = Memcpy->getParent()->getParent();
            const TargetTransformInfo &TTI =
                AM.getResult<TargetIRAnalysis>(*ParentFunc);
            expandMemCpyAsLoop(Memcpy, TTI);
            Memcpy->eraseFromParent();
        }
        else if (MemMoveInst *Memmove = dyn_cast<MemMoveInst>(MemCall)) {
#if LLVM_VERSION_MAJOR >= 17
            Function *ParentFunc = Memmove->getParent()->getParent();
            const TargetTransformInfo &TTI =
                AM.getResult<TargetIRAnalysis>(*ParentFunc);
            expandMemMoveAsLoop(Memmove, TTI);
#else
            expandMemMoveAsLoop(Memmove);
#endif
            Memmove->eraseFromParent();
        }
        else if (MemSetInst *Memset = dyn_cast<MemSetInst>(MemCall)) {
            expandMemSetAsLoop(Memset);
            Memset->eraseFromParent();
        }
    }

    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();

    return PA;
}

bool
aot_check_simd_compatibility(LLVMTargetMachineRef target_machine)
{
#if WASM_ENABLE_SIMD != 0
    if (!target_machine) {
        return false;
    }

    /* Query the target machine that will actually be used for code
       generation. Re-deriving a subtarget from the CPU name alone would
       drop the feature string, and a CPU name unknown to this LLVM version
       falls back to "generic", which advertises no SIMD feature even when
       the host does support it. */
    const llvm::TargetMachine *targetMachine =
        reinterpret_cast<llvm::TargetMachine *>(target_machine);
    const llvm::MCSubtargetInfo *subTargetInfo =
        targetMachine->getMCSubtargetInfo();
    if (subTargetInfo == nullptr) {
        return false;
    }

    switch (targetMachine->getTargetTriple().getArch()) {
        case llvm::Triple::x86_64:
            return subTargetInfo->checkFeatures("+sse4.1");
        case llvm::Triple::aarch64:
        case llvm::Triple::aarch64_be:
            return subTargetInfo->checkFeatures("+neon");
        case llvm::Triple::arc:
            return true;
        default:
            return false;
    }
#else
    (void)target_machine;
    return true;
#endif /* WASM_ENABLE_SIMD */
}

bool
aot_apply_llvm_new_pass_manager(AOTCompContext *comp_ctx, LLVMModuleRef module)
{
    TargetMachine *TM =
        reinterpret_cast<TargetMachine *>(comp_ctx->target_machine);
    PipelineTuningOptions PTO;
    PTO.LoopVectorization = true;
    PTO.SLPVectorization = true;
    PTO.LoopUnrolling = true;

#if LLVM_VERSION_MAJOR >= 16
    Optional<PGOOptions> PGO = std::nullopt;
#else
    Optional<PGOOptions> PGO = llvm::None;
#endif

    if (comp_ctx->enable_llvm_pgo) {
        /* Disable static counter allocation for value profiler,
           it will be allocated by runtime */
        const char *argv[] = { "", "-vp-static-alloc=false" };
        cl::ParseCommandLineOptions(2, argv);
#if LLVM_VERSION_MAJOR < 17
        PGO = PGOOptions("", "", "", PGOOptions::IRInstr);
#elif LLVM_VERSION_MAJOR < 22
        auto FS = vfs::getRealFileSystem();
        PGO = PGOOptions("", "", "", "", FS, PGOOptions::IRInstr);
#else
        PGO = PGOOptions("", "", "", "", PGOOptions::IRInstr);
#endif
    }
    else if (comp_ctx->use_prof_file) {
#if LLVM_VERSION_MAJOR < 17
        PGO = PGOOptions(comp_ctx->use_prof_file, "", "", PGOOptions::IRUse);
#elif LLVM_VERSION_MAJOR < 22
        auto FS = vfs::getRealFileSystem();
        PGO = PGOOptions(comp_ctx->use_prof_file, "", "", "", FS,
                         PGOOptions::IRUse);
#else
        PGO =
            PGOOptions(comp_ctx->use_prof_file, "", "", "", PGOOptions::IRUse);
#endif
    }

#ifdef DEBUG_PASS
    PassInstrumentationCallbacks PIC;
    PassBuilder PB(TM, PTO, PGO, &PIC);
#else
#if LLVM_VERSION_MAJOR == 12
    PassBuilder PB(false, TM, PTO, PGO);
#else
    PassBuilder PB(TM, PTO, PGO);
#endif
#endif

    /* Register all the basic analyses with the managers */
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    /* Register the target library analysis directly and give it a
       customized preset TLI */
    std::unique_ptr<TargetLibraryInfoImpl> TLII(
        new TargetLibraryInfoImpl(Triple(TM->getTargetTriple())));
    FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

    /* Register the AA manager first so that our version is the one used */
    AAManager AA = PB.buildDefaultAAPipeline();
    FAM.registerPass([&] { return std::move(AA); });

#ifdef DEBUG_PASS
    StandardInstrumentations SI(true, false);
    SI.registerCallbacks(PIC, &FAM);
#endif

    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

#if LLVM_VERSION_MAJOR <= 13
    PassBuilder::OptimizationLevel OL;

    switch (comp_ctx->opt_level) {
        case 0:
            OL = PassBuilder::OptimizationLevel::O0;
            break;
        case 1:
            OL = PassBuilder::OptimizationLevel::O1;
            break;
        case 2:
            OL = PassBuilder::OptimizationLevel::O2;
            break;
        case 3:
        default:
            OL = PassBuilder::OptimizationLevel::O3;
            break;
    }
#else
    OptimizationLevel OL;

    switch (comp_ctx->opt_level) {
        case 0:
            OL = OptimizationLevel::O0;
            break;
        case 1:
            OL = OptimizationLevel::O1;
            break;
        case 2:
            OL = OptimizationLevel::O2;
            break;
        case 3:
        default:
            OL = OptimizationLevel::O3;
            break;
    }
#endif /* end of LLVM_VERSION_MAJOR */

    bool disable_llvm_lto = comp_ctx->disable_llvm_lto;
#if WASM_ENABLE_SPEC_TEST != 0
    disable_llvm_lto = true;
#endif

    Module *M = reinterpret_cast<Module *>(module);
    if (disable_llvm_lto) {
        for (Function &F : *M) {
            F.addFnAttr("disable-tail-calls", "true");
        }
    }

    ModulePassManager MPM;

    if (comp_ctx->is_jit_mode) {
#if LLVM_VERSION_MAJOR >= 18
#define INSTCOMBINE "instcombine<no-verify-fixpoint>"
#else
#define INSTCOMBINE "instcombine"
#endif
        const char *Passes =
            "loop-vectorize,slp-vectorizer,"
            "load-store-vectorizer,vector-combine,"
            "mem2reg," INSTCOMBINE ",simplifycfg,jump-threading,indvars";
        if (auto error = PB.parsePassPipeline(MPM, Passes)) {
            aot_set_last_error(toString(std::move(error)).c_str());
            return false;
        }
    }
    else {
        FunctionPassManager FPM;

        /* Apply Vectorize related passes for AOT mode */
        FPM.addPass(LoopVectorizePass());
        FPM.addPass(SLPVectorizerPass());
        FPM.addPass(LoadStoreVectorizerPass());
        FPM.addPass(VectorCombinePass());

        if (comp_ctx->enable_llvm_pgo || comp_ctx->use_prof_file) {
            /* LICM pass: loop invariant code motion, attempting to remove
               as much code from the body of a loop as possible. Experiments
               show it is good to enable it when pgo is enabled. */
#if LLVM_VERSION_MAJOR >= 15
            LICMOptions licm_opt;
            FPM.addPass(
                createFunctionToLoopPassAdaptor(LICMPass(licm_opt), true));
#else
            FPM.addPass(createFunctionToLoopPassAdaptor(LICMPass(), true));
#endif
        }

        /*
        FPM.addPass(createFunctionToLoopPassAdaptor(LoopRotatePass()));
        FPM.addPass(createFunctionToLoopPassAdaptor(SimpleLoopUnswitchPass()));
        */

        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

        if (comp_ctx->llvm_passes) {
            if (auto error = PB.parsePassPipeline(MPM, comp_ctx->llvm_passes)) {
                aot_set_last_error(toString(std::move(error)).c_str());
                return false;
            }
        }

        if (
#if LLVM_VERSION_MAJOR <= 13
            PassBuilder::OptimizationLevel::O0 == OL
#else
            OptimizationLevel::O0 == OL
#endif
        ) {
            MPM.addPass(PB.buildO0DefaultPipeline(OL));
        }
        else {
            if (!disable_llvm_lto) {
                /* Apply LTO for AOT mode */
                if (comp_ctx->comp_data->func_count >= 10
                    || comp_ctx->enable_llvm_pgo || comp_ctx->use_prof_file)
                    /* Add the pre-link optimizations if the func count
                       is large enough or PGO is enabled */
                    MPM.addPass(PB.buildLTOPreLinkDefaultPipeline(OL));
                else
                    MPM.addPass(PB.buildLTODefaultPipeline(OL, NULL));
            }
            else {
                MPM.addPass(PB.buildPerModuleDefaultPipeline(OL));
            }
        }

        /* Run specific passes for AOT indirect mode in last since general
            optimization may create some intrinsic function calls like
            llvm.memset, so let's remove these function calls here. */
        if (comp_ctx->is_indirect_mode) {
            FunctionPassManager FPM1;
            FPM1.addPass(ExpandMemoryOpPass());
            MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM1)));
        }
    }

    MPM.run(*M, MAM);
    return true;
}

char *
aot_compress_aot_func_names(AOTCompContext *comp_ctx, uint32 *p_size)
{
    std::vector<std::string> NameStrs;
    std::string Result;
    char buf[32], *compressed_str;
    uint32 compressed_str_len, i;

    for (i = 0; i < comp_ctx->func_ctx_count; i++) {
        snprintf(buf, sizeof(buf), "%s%d", AOT_FUNC_PREFIX, i);
        std::string str(buf);
        NameStrs.push_back(str);
    }

#if LLVM_VERSION_MAJOR < 18
#define collectGlobalObjectNameStrings collectPGOFuncNameStrings
#endif
    if (collectGlobalObjectNameStrings(NameStrs, true, Result)) {
        aot_set_last_error("collect pgo func name strings failed");
        return NULL;
    }

    compressed_str_len = (uint32)Result.size();
    if (!(compressed_str = (char *)wasm_runtime_malloc(compressed_str_len))) {
        aot_set_last_error("allocate memory failed");
        return NULL;
    }

    bh_memcpy_s(compressed_str, compressed_str_len, Result.c_str(),
                compressed_str_len);
    *p_size = compressed_str_len;
    return compressed_str;
}
