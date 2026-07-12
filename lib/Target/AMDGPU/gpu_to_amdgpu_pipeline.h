#pragma once

#include <mlir/Pass/PassManager.h>

namespace causalflow::avelang::ir {
class IRContext;
}

namespace causalflow::avelang::target::amdgpu {

struct AMDGPUToLLVMPipelineOptions {
    std::string chipset = "gfx90a";
    std::string triple = "amdgcn-amd-amdhsa";
    int optimization_level = 3;
    int num_warps = -1;
    bool validate_invariants = false;
    bool use_bare_ptr_memref_call_conv = true;
    std::string target_features = "";
};

void BuildLowerToAMDGPUPassPipeline(mlir::OpPassManager &pm,
                                    const AMDGPUToLLVMPipelineOptions &options);
} // namespace causalflow::avelang::target::amdgpu
