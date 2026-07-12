#include "nvvm_backend.h"
#include "Target/GPU/lower_to_llvm.h"
#include "gpu_to_nvvm_pipeline.h"
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <mutex>
#include <sstream>

namespace causalflow::avelang::target::nvvm {

void NVVMBackend::buildLoweringPipeline(
    mlir::OpPassManager &pm,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    NVVMToLLVMPipelineOptions nvvmOptions;
    nvvmOptions.chipset = options.chipset.str();
    nvvmOptions.triple = options.triple.str();
    nvvmOptions.optimization_level = options.optimization_level;
    nvvmOptions.num_warps = options.num_warps;
    nvvmOptions.validate_invariants = options.validate_invariants;
    nvvmOptions.use_bare_ptr_memref_call_conv =
        options.use_bare_ptr_memref_call_conv;

    BuildLowerToNVVMPassPipeline(pm, nvvmOptions);
}

bool NVVMBackend::supportsTriple(llvm::StringRef triple) const {
    return triple.contains("nvptx") || triple.contains("nvidia");
}

std::string NVVMBackend::getName() const { return "NVVM"; }

void NVVMBackend::EnsureInitialized() {
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
    });
}

llvm::Expected<std::string> NVVMBackend::generateBinary(
    llvm::Module &module,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    EnsureInitialized();

    // Get the target triple
    std::string targetTripleStr = options.triple.str();
    if (targetTripleStr.empty()) {
        targetTripleStr = "nvptx64-nvidia-cuda";
    }
    llvm::Triple targetTriple(targetTripleStr);

    // Set up the target machine
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to lookup NVPTX target: " +
                                           error);
    }

    llvm::TargetOptions targetOptions;
    auto targetMachine =
        std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
            targetTriple, options.chipset.str(), "", targetOptions,
            llvm::Reloc::PIC_, std::nullopt,
            options.optimization_level == 3 ? llvm::CodeGenOptLevel::Aggressive
            : options.optimization_level == 2 ? llvm::CodeGenOptLevel::Default
            : options.optimization_level == 1 ? llvm::CodeGenOptLevel::Less
                                              : llvm::CodeGenOptLevel::None,
            false));
    if (!targetMachine) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to create NVPTX target machine");
    }

    // Set up the module with the target data layout
    module.setTargetTriple(targetTriple);
    module.setDataLayout(targetMachine->createDataLayout());

    // Generate PTX assembly using raw_svector_ostream
    llvm::SmallVector<char, 1024> ptxBuffer;
    llvm::raw_svector_ostream ptxStream(ptxBuffer);

    llvm::legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(
            passManager, ptxStream, nullptr,
            llvm::CodeGenFileType::AssemblyFile)) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Target machine cannot emit PTX assembly");
    }

    passManager.run(module);
    std::string ptxAssembly = ptxStream.str().str();

    if (ptxAssembly.empty()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Generated PTX assembly is empty");
    }

    return ptxAssembly;
}

llvm::Expected<std::string> NVVMBackend::generateAssembly(
    llvm::Module &module,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    // For NVVM, assembly is the same as binary (PTX)
    return generateBinary(module, options);
}

std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateNVVMBackend() {
    return std::make_unique<NVVMBackend>();
}

} // namespace causalflow::avelang::target::nvvm
