#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::ir {
class IRContext;
}

namespace causalflow::avelang::target::gpu {

/// GPU compilation options
struct GPUCompilationOptions {
    /// Target chipset (e.g., "gfx90a", "sm_80")
    llvm::StringRef chipset = "gfx90a";

    /// Target triple (e.g., "amdgcn-amd-amdhsa", "nvptx64-nvidia-cuda")
    llvm::StringRef triple = "amdgcn-amd-amdhsa";

    /// Optimization level (0-3)
    unsigned optimization_level = 3;

    /// Number of warps (or waves) requested by the frontend. -1 means unset.
    int num_warps = -1;

    /// Run compile-time data-flow invariant proofs.
    bool validate_invariants = false;

    /// Use bare pointer calling convention for memref
    bool use_bare_ptr_memref_call_conv = true;

    /// Target features string (e.g., "+wavefrontsize64,-sramecc,+xnack")
    std::string target_features = "";
};

// Lower MLIR to LLVM IR
class LowerToLLVM {
  public:
    explicit LowerToLLVM(causalflow::avelang::ir::IRContext *ir_context);
    ~LowerToLLVM();

    std::unique_ptr<::llvm::Module>
    compile(mlir::ModuleOp module, ::llvm::LLVMContext &llvmContext,
            const GPUCompilationOptions &options = {});

    mlir::MLIRContext *getContext();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace causalflow::avelang::target::gpu
