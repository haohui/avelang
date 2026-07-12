#pragma once

#include <mlir/Pass/PassManager.h>

namespace causalflow::avelang::ir {
class IRContext;
}

namespace causalflow::avelang::target::nvvm {

struct NVVMToLLVMPipelineOptions {
    std::string chipset = "sm_80";
    std::string triple = "nvptx64-nvidia-cuda";
    int optimization_level = 2;
    int num_warps = -1;
    bool validate_invariants = false;
    bool use_bare_ptr_memref_call_conv = true;
};

void BuildLowerToNVVMPassPipeline(mlir::OpPassManager &pm,
                                  const NVVMToLLVMPipelineOptions &options);
} // namespace causalflow::avelang::target::nvvm
