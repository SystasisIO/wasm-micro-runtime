#include "aot_jit_profile.h"

#include <llvm/Config/llvm-config.h>
#include <algorithm>

#if LLVM_VERSION_MAJOR >= 22
#include <limits>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/DebugInfo/DWARF/DWARFDebugLine.h>
#include <llvm/DebugInfo/DWARF/DWARFUnit.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <new>
#include <vector>

using namespace llvm;

namespace
{
/* The compact map, adapter buffers and plugin records belong to the existing
   WAMR allocation owner. LLVM's decoder/IR/object scratch heap is not a claimed
   part of that ceiling, just as for ordinary LLVM JIT compilation. */
template <class T> struct ProfileAllocator {
    using value_type = T;
    ProfileAllocator() = default;
    template <class U> ProfileAllocator(const ProfileAllocator<U> &) {}
    T *allocate(size_t count)
    {
        if (count > UINT32_MAX / sizeof(T))
            throw std::bad_alloc();
        auto *p = wasm_runtime_malloc(static_cast<uint32>(count * sizeof(T)));
        if (!p)
            throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, size_t) { wasm_runtime_free(p); }
    template <class U> bool operator==(const ProfileAllocator<U> &) const
    {
        return true;
    }
};
template <class T> using ProfileVector = std::vector<T, ProfileAllocator<T>>;

struct GuestFunction {
    char name[64];
};
struct NativeRange {
    uint64_t begin;
    uint64_t end;
};
struct DebugBlock {
    jitlink::Block *block;
    uint64_t address;
};
struct DebugSection {
    StringRef name;
    ProfileVector<DebugBlock> blocks;
    ProfileVector<char> bytes;
};

Error profile_error(const char *text)
{
    return make_error<StringError>(text, inconvertibleErrorCode());
}

StringRef debug_section_name(StringRef name)
{
    if (name.consume_front("__DWARF,__debug_"))
        return name;
    if (name.consume_front(".debug_"))
        return name;
    return {};
}
} // namespace

struct AOTJitProfile {
    std::unique_ptr<DIBuilder> builder;
    DIFile *file = nullptr;
    ProfileVector<GuestFunction> functions;
    ProfileVector<AOTJitProfileInterval> pending;
    ProfileVector<AOTJitProfileInterval> intervals;
    WASMLoadBudget budget;
    uint64_t wasm_size = 0;
    bool linked = false;
    bool published = false;
    bool failed = false;

    void clear()
    {
        ProfileVector<AOTJitProfileInterval>().swap(pending);
        ProfileVector<AOTJitProfileInterval>().swap(intervals);
        linked = false;
        published = false;
    }
};

namespace
{
class ProfilePlugin final : public orc::ObjectLinkingLayer::Plugin
{
  public:
    explicit ProfilePlugin(AOTJitProfile &profile) : profile(profile) {}

    void modifyPassConfig(orc::MaterializationResponsibility &, jitlink::LinkGraph &,
                          jitlink::PassConfiguration &passes) override
    {
        passes.PrePrunePasses.push_back(
            [this](jitlink::LinkGraph &graph) { return preserve(graph); });
        passes.PostFixupPasses.push_back(
            [this](jitlink::LinkGraph &graph) { return extract(graph); });
    }
    Error notifyEmitted(orc::MaterializationResponsibility &) override
    {
        if (!profile.budget.check(profile.budget.context))
            return profile_error("JIT profiling preparation canceled");
        if (profile.pending.empty())
            return profile_error("JIT instruction profile has no native rows");
        profile.intervals.swap(profile.pending);
        profile.linked = true;
        ProfileVector<DebugSection>().swap(sections);
        return Error::success();
    }
    Error notifyFailed(orc::MaterializationResponsibility &) override
    {
        profile.failed = true;
        profile.clear();
        ProfileVector<DebugSection>().swap(sections);
        return Error::success();
    }
    Error notifyRemovingResources(orc::JITDylib &, orc::ResourceKey) override
    {
        profile.clear();
        return Error::success();
    }
    void notifyTransferringResources(orc::JITDylib &, orc::ResourceKey,
                                     orc::ResourceKey) override
    {
    }

  private:
    Error preserve(jitlink::LinkGraph &graph)
    {
        try {
            if (!profile.budget.check(profile.budget.context))
                return profile_error("JIT profiling preparation canceled");
            if (profile.linked || !profile.pending.empty())
                return profile_error(
                    "JIT instruction profile requires one eager object");
            /* Native LLVM DebugInfoSupport currently accepts ELF only. This
               adapter preserves the Mach-O NoAlloc debug blocks as well; all
               relocation and DWARF decoding remain LLVM's responsibility. */
            for (auto &section : graph.sections()) {
                if (!profile.budget.check(profile.budget.context))
                    return profile_error("JIT profiling preparation canceled");
                auto name = debug_section_name(section.getName());
                if (name.empty())
                    continue;
                if (section.getMemLifetime() != orc::MemLifetime::NoAlloc)
                    return profile_error(
                        "JIT debug section unexpectedly requires mapping");
                DebugSection saved;
                saved.name = name;
                for (auto *block : section.blocks()) {
                    if (block->isZeroFill())
                        return profile_error("JIT debug section is zero filled");
                    saved.blocks.push_back({block, block->getAddress().getValue()});
                    graph.addAnonymousSymbol(*block, 0, block->getSize(), false, true);
                }
                sections.push_back(std::move(saved));
            }
            if (sections.empty())
                return profile_error("JIT object contains no DWARF sections");
            /* Keep targets referenced by debug relocations alive before graph
               pruning. This is LLVM's debug-preservation policy, not a second
               symbol resolver or relocation implementation. */
            for (auto &section : sections)
                for (auto &saved : section.blocks)
                    for (auto &edge : saved.block->edges())
                        edge.getTarget().setLive(true);
            return Error::success();
        } catch (const std::bad_alloc &) {
            profile.budget.allocation_failed(profile.budget.context);
            return profile_error("JIT debug preservation allocation failed");
        }
    }

    Error extract(jitlink::LinkGraph &graph)
    {
        try {
            if (!profile.budget.check(profile.budget.context))
                return profile_error("JIT profiling preparation canceled");
            ProfileVector<NativeRange> ranges;
            for (auto *symbol : graph.defined_symbols()) {
                if (!symbol->hasName() || !symbol->isCallable() || !symbol->getSize())
                    continue;
                auto name = *symbol->getName();
                if (graph.getTargetTriple().isOSDarwin())
                    name.consume_front("_");
                const bool guest =
                    llvm::any_of(profile.functions, [&](const GuestFunction &function) {
                        return name == function.name;
                    });
                if (!guest)
                    continue;
                const uint64_t begin = symbol->getAddress().getValue();
                if (symbol->getSize() > UINTPTR_MAX - begin)
                    return profile_error("JIT function extent overflow");
                ranges.push_back({begin, begin + symbol->getSize()});
            }
            llvm::sort(ranges, [](auto &left, auto &right) {
                return left.begin < right.begin;
            });
            for (size_t i = 1; i < ranges.size(); ++i)
                if (ranges[i - 1].end > ranges[i].begin)
                    return profile_error("JIT guest function extents overlap");
            if (ranges.empty())
                return profile_error("JIT profile has no linked guest functions");

            StringMap<std::unique_ptr<MemoryBuffer>> buffers;
            for (auto &section : sections) {
                if (!profile.budget.check(profile.budget.context))
                    return profile_error("JIT profiling preparation canceled");
                llvm::sort(section.blocks, [](auto &left, auto &right) {
                    return left.address < right.address;
                });
                uint64_t cursor = 0;
                bool first = true;
                for (auto &saved : section.blocks) {
                    auto &block = *saved.block;
                    /* NoAlloc blocks retain object addresses. Require complete
                       section coverage instead of guessing section bases or
                       assuming executable blocks stay contiguous. */
                    if (saved.address != block.getAddress().getValue() ||
                        (!first && saved.address != cursor))
                        return profile_error("JIT DWARF section layout changed");
                    if (block.getSize() > UINT32_MAX - section.bytes.size())
                        return profile_error("JIT DWARF adapter extent overflow");
                    auto content = block.getContent();
                    section.bytes.insert(section.bytes.end(), content.begin(),
                                         content.end());
                    cursor = saved.address + block.getSize();
                    first = false;
                }
                std::string name = (Twine("debug_") + section.name).str();
                if (buffers.count(name))
                    return profile_error("Duplicate JIT DWARF section");
                buffers[name] = MemoryBuffer::getMemBuffer(
                    StringRef(section.bytes.data(), section.bytes.size()), name, false);
            }
            bool malformed = false;
            auto diagnostic = [&](Error error) {
                malformed = true;
                consumeError(std::move(error));
            };
            auto dwarf = DWARFContext::create(
                buffers, graph.getPointerSize(),
                graph.getEndianness() == endianness::little, diagnostic, diagnostic);
            for (auto &unit : dwarf->compile_units()) {
                if (!profile.budget.check(profile.budget.context))
                    return profile_error("JIT profiling preparation canceled");
                auto table = dwarf->getLineTableForUnit(unit.get(), diagnostic);
                if (!table)
                    return table.takeError();
                if (!*table)
                    return profile_error("JIT DWARF line table is missing");
                const auto &rows = (*table)->Rows;
                for (size_t i = 0; i + 1 < rows.size(); ++i) {
                    const auto &row = rows[i];
                    const auto &next = rows[i + 1];
                    if (!row.Line || row.EndSequence)
                        continue;
                    if (row.Line > profile.wasm_size)
                        return profile_error("JIT DWARF location exceeds the original WASM file");
                    const uint64_t begin = row.Address.Address;
                    const uint64_t end = next.Address.Address;
                    if (begin >= end)
                        continue;
                    auto range =
                        std::upper_bound(ranges.begin(), ranges.end(), begin,
                                         [](uint64_t pc, const NativeRange &item) {
                                             return pc < item.begin;
                                         });
                    if (range == ranges.begin())
                        continue;
                    --range;
                    /* Neither padding, helpers, prologue line zero nor another
                       discontiguous allocation inherits a preceding row. */
                    if (begin < range->begin || begin >= range->end)
                        continue;
                    profile.pending.push_back(
                        {static_cast<uintptr_t>(begin),
                         static_cast<uintptr_t>(std::min(end, range->end)),
                         row.Line - 1});
                }
            }
            if (malformed)
                return profile_error("JIT DWARF decoding failed");
            llvm::sort(profile.pending, [](auto &left, auto &right) {
                return left.begin < right.begin;
            });
            for (size_t i = 1; i < profile.pending.size(); ++i)
                if (profile.pending[i - 1].end > profile.pending[i].begin)
                    return profile_error("JIT instruction intervals overlap");
            if (profile.pending.empty())
                return profile_error(
                    "JIT DWARF has no attributable instruction ranges");
            return Error::success();
        } catch (const std::bad_alloc &) {
            profile.budget.allocation_failed(profile.budget.context);
            return profile_error("JIT instruction map allocation failed");
        }
    }

    AOTJitProfile &profile;
    ProfileVector<DebugSection> sections;
};

DISubprogram *function_scope(AOTJitProfile &profile, Function &function)
{
    auto type = profile.builder->createSubroutineType(
        profile.builder->getOrCreateTypeArray({}));
    auto *scope = profile.builder->createFunction(
        profile.file, function.getName(), function.getName(), profile.file, 0, type, 0,
        DINode::FlagPrototyped,
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
    function.setSubprogram(scope);
    return scope;
}
} // namespace
#endif

bool aot_jit_profile_enable(AOTCompContext *comp_ctx)
{
#if LLVM_VERSION_MAJOR >= 22 && WASM_ENABLE_DEBUG_AOT == 0
    if (!comp_ctx || !comp_ctx->is_jit_mode ||
        !comp_ctx->jit_compile_on_caller_thread || !comp_ctx->module ||
        comp_ctx->jit_instruction_profile || !comp_ctx->func_ctx_count ||
        !comp_ctx->jit_load_budget.check ||
        !comp_ctx->jit_load_budget.allocation_failed) {
        aot_set_last_error(
            "Instruction profile requires a fresh bounded caller-thread JIT");
        return false;
    }
    if (!comp_ctx->jit_load_budget.check(comp_ctx->jit_load_budget.context)) {
        aot_set_last_error("JIT profiling preparation canceled or unavailable");
        return false;
    }
    for (uint32 i = 0; i < comp_ctx->func_ctx_count; ++i) {
        if (!comp_ctx->func_ctxes[i]->block_stack.block_list_head) {
            aot_set_last_error("Instruction profiling must precede opcode translation");
            return false;
        }
    }
    auto *storage = wasm_runtime_malloc(sizeof(AOTJitProfile));
    if (!storage) {
        comp_ctx->jit_load_budget.allocation_failed(comp_ctx->jit_load_budget.context);
        aot_set_last_error("JIT instruction profile allocation failed");
        return false;
    }
    auto *profile = new (storage) AOTJitProfile();
    comp_ctx->jit_instruction_profile = profile;
    profile->budget = comp_ctx->jit_load_budget;
    profile->wasm_size = comp_ctx->comp_data->wasm_module->load_size;
    try {
        auto &module = *unwrap(comp_ctx->module);
        module.addModuleFlag(Module::Warning, "Debug Info Version",
                             DEBUG_METADATA_VERSION);
        module.addModuleFlag(Module::Warning, "Dwarf Version", 4);
        profile->builder = std::make_unique<DIBuilder>(module);
        profile->file = profile->builder->createFile("module.wasm", ".");
        profile->builder->createCompileUnit(
            dwarf::DW_LANG_C, profile->file, "WAMR WASM instruction offsets", true, "",
            0, "", DICompileUnit::FullDebug, 0, false, true,
            DICompileUnit::DebugNameTableKind::None);
        for (uint32 i = 0; i < comp_ctx->func_ctx_count; ++i) {
            if (!profile->budget.check(profile->budget.context)) {
                profile->failed = true;
                aot_set_last_error("JIT profiling preparation canceled or unavailable");
                return false;
            }
            auto *function = unwrap<Function>(comp_ctx->func_ctxes[i]->func);
            if (function->getName().size() >= sizeof(GuestFunction::name))
                throw std::bad_alloc();
            GuestFunction saved{};
            memcpy(saved.name, function->getName().data(), function->getName().size());
            profile->functions.push_back(saved);
        }
        for (auto &function : module) {
            if (!profile->budget.check(profile->budget.context)) {
                profile->failed = true;
                aot_set_last_error("JIT profiling preparation canceled or unavailable");
                return false;
            }
            if (!function.isDeclaration())
                function_scope(*profile, function);
        }
        auto *base = reinterpret_cast<orc::ObjectLayer *>(
            LLVMOrcLLLazyJITGetObjLinkingLayer(comp_ctx->orc_jit));
        auto *layer = dyn_cast<orc::ObjectLinkingLayer>(base);
        if (!layer) {
            profile->failed = true;
            aot_set_last_error(
                "Instruction profiling requires the bounded JITLink object layer");
            return false;
        }
        layer->addPlugin(std::allocate_shared<ProfilePlugin>(
            ProfileAllocator<ProfilePlugin>{}, *profile));
        return true;
    } catch (const std::bad_alloc &) {
        profile->failed = true;
        profile->budget.allocation_failed(profile->budget.context);
        aot_set_last_error("JIT instruction profile metadata allocation failed");
        /* The caller must discard this failed compiler context, not compile
           partial metadata or publish a function-only substitute. */
        return false;
    }
#else
    (void)comp_ctx;
    aot_set_last_error(
        "Instruction profiling requires LLVM 22 and no source-DWARF translator");
    return false;
#endif
}

bool aot_jit_profile_location(AOTCompContext *comp_ctx, AOTFuncContext *func_ctx,
                              const uint8 *opcode)
{
#if LLVM_VERSION_MAJOR >= 22
    auto *profile = comp_ctx->jit_instruction_profile;
    if (!profile)
        return true;
    if (profile->failed) {
        aot_set_last_error("Cannot reuse failed JIT profiling metadata");
        return false;
    }
    try {
        if (!profile->budget.check(profile->budget.context)) {
            profile->failed = true;
            aot_set_last_error("JIT profiling preparation canceled or unavailable");
            return false;
        }
        unsigned line = 0;
        if (opcode) {
            const auto *module = comp_ctx->comp_data->wasm_module;
            const auto address = reinterpret_cast<uintptr_t>(opcode);
            const auto base = reinterpret_cast<uintptr_t>(module->load_addr);
            if (!base || address < base || address - base >= module->load_size ||
                address - base >= UINT32_MAX) {
                profile->failed = true;
                aot_set_last_error(
                    "JIT instruction location is outside the original WASM file");
                return false;
            }
            line = static_cast<unsigned>(address - base + 1);
        }
        auto *function = unwrap<Function>(func_ctx->func);
        unwrap(comp_ctx->builder)
            ->SetCurrentDebugLocation(DILocation::get(function->getContext(), line, 0,
                                                      function->getSubprogram()));
    } catch (const std::bad_alloc &) {
        profile->failed = true;
        profile->budget.allocation_failed(profile->budget.context);
        aot_set_last_error("JIT instruction location allocation failed");
        return false;
    }
#else
    (void)comp_ctx;
    (void)func_ctx;
    (void)opcode;
#endif
    return true;
}

bool aot_jit_profile_finalize(AOTCompContext *comp_ctx)
{
#if LLVM_VERSION_MAJOR >= 22
    auto *profile = comp_ctx->jit_instruction_profile;
    if (!profile)
        return true;
    if (profile->failed || !profile->builder ||
        !profile->budget.check(profile->budget.context)) {
        profile->failed = true;
        aot_set_last_error("JIT profiling metadata is failed or unavailable");
        return false;
    }
    try {
        auto &module = *unwrap(comp_ctx->module);
        for (auto &function : module) {
            if (!profile->budget.check(profile->budget.context)) {
                profile->failed = true;
                aot_set_last_error("JIT profiling preparation canceled or unavailable");
                return false;
            }
            if (function.isDeclaration())
                continue;
            auto *scope = function.getSubprogram();
            if (!scope)
                scope = function_scope(*profile, function);
            const auto unknown = DILocation::get(function.getContext(), 0, 0, scope);
            for (auto &block : function)
                for (auto &instruction : block)
                    if (!instruction.getDebugLoc())
                        instruction.setDebugLoc(unknown);
        }
        profile->builder->finalize();
        profile->builder.reset();
    } catch (const std::bad_alloc &) {
        profile->failed = true;
        profile->budget.allocation_failed(profile->budget.context);
        aot_set_last_error("JIT profiling metadata finalization allocation failed");
        return false;
    }
#else
    (void)comp_ctx;
#endif
    return true;
}

void aot_jit_profile_discard_metadata(AOTCompContext *comp_ctx)
{
#if LLVM_VERSION_MAJOR >= 22
    if (auto *profile = comp_ctx->jit_instruction_profile)
        profile->builder.reset();
#else
    (void)comp_ctx;
#endif
}

bool aot_jit_profile_publish(AOTCompContext *comp_ctx)
{
#if LLVM_VERSION_MAJOR >= 22
    auto *profile = comp_ctx->jit_instruction_profile;
    if (!profile)
        return true;
    if (profile->failed || profile->builder || !profile->linked ||
        profile->intervals.empty() ||
        !profile->budget.check(profile->budget.context)) {
        aot_set_last_error("JIT instruction profile is incomplete or unavailable");
        return false;
    }
    profile->published = true;
#else
    (void)comp_ctx;
#endif
    return true;
}

void aot_jit_profile_destroy(AOTCompContext *comp_ctx)
{
#if LLVM_VERSION_MAJOR >= 22
    if (auto *profile = comp_ctx->jit_instruction_profile) {
        profile->~AOTJitProfile();
        wasm_runtime_free(profile);
        comp_ctx->jit_instruction_profile = nullptr;
    }
#else
    (void)comp_ctx;
#endif
}

const AOTJitProfileInterval *aot_jit_profile_intervals(const AOTCompContext *comp_ctx,
                                                       uint32 *count)
{
    if (!count)
        return nullptr;
    *count = 0;
#if LLVM_VERSION_MAJOR >= 22
    const auto *profile = comp_ctx ? comp_ctx->jit_instruction_profile : nullptr;
    if (profile && profile->published) {
        *count = static_cast<uint32>(profile->intervals.size());
        return profile->intervals.data();
    }
#else
    (void)comp_ctx;
#endif
    return nullptr;
}

uint32 aot_jit_profile_translate(const AOTCompContext *comp_ctx, uintptr_t pc)
{
    uint32 count;
    const auto *rows = aot_jit_profile_intervals(comp_ctx, &count);
    if (!count)
        return UINT32_MAX;
    const auto *found =
        std::upper_bound(rows, rows + count, pc,
                         [](uintptr_t address, const AOTJitProfileInterval &row) {
                             return address < row.begin;
                         });
    if (found == rows || pc >= (--found)->end)
        return UINT32_MAX;
    return found->wasm_offset;
}
