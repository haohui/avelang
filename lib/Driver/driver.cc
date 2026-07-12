#include "driver.h"
#include "avelang/config.h"

#include "AST/ast_context.h"
#include "Frontend/avelang_parser.h"
#include "IR/ir_context.h"
#include "IR/mlir_generator.h"
#include "Target/GPU/gpu_backend.h"
#include "Target/GPU/lower_to_llvm.h"

#include <clang/Basic/Diagnostic.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

using namespace llvm;

namespace causalflow::avelang::driver {

//===----------------------------------------------------------------------===//
// Static helper functions
//===----------------------------------------------------------------------===//

static target::gpu::GPUCompilationOptions
toGPUOptions(const CompilationOptions &Options) {
    target::gpu::GPUCompilationOptions GPUOptions;
    GPUOptions.chipset = Options.TargetChipset;
    GPUOptions.triple = Options.TargetTriple;
    GPUOptions.optimization_level = Options.OptLevel;
    GPUOptions.num_warps = Options.NumWarps;
    GPUOptions.validate_invariants = Options.ValidateInvariants;
    GPUOptions.use_bare_ptr_memref_call_conv = Options.UseBarePointerCallConv;
    return GPUOptions;
}

static Expected<std::unique_ptr<target::gpu::GPUBackendInterface>>
getBackendForTriple(StringRef Triple) {
    target::gpu::GPUBackendRegistry &registry =
        target::gpu::GPUBackendRegistry::getInstance();
    auto backend = registry.createBackendForTriple(Triple);
    if (!backend) {
        return createStringError(std::make_error_code(std::errc::not_supported),
                                 "No backend available for target: " +
                                     Triple.str());
    }
    return backend;
}

//===----------------------------------------------------------------------===//
// Driver implementation
//===----------------------------------------------------------------------===//

class Driver::Impl {
  public:
    Impl(llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagMgr,
         ::llvm::LLVMContext &llvmCtx)
        : DiagnosticManager(diagMgr), LLVMContext(&llvmCtx) {
        ASTContext = new ast::ASTContext();
        IRContext = ir::IRContext::Create();
    }

    IntrusiveRefCntPtr<ast::ASTContext> ASTContext;
    IntrusiveRefCntPtr<basic::DiagnosticManager> DiagnosticManager;
    ::llvm::LLVMContext *LLVMContext;
    std::unique_ptr<ir::IRContext> IRContext;
};

Driver::Driver(llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> DiagMgr,
               ::llvm::LLVMContext &LLVMCtx)
    : Impl(std::make_unique<class Impl>(DiagMgr, LLVMCtx)) {}

Driver::~Driver() = default;

//===----------------------------------------------------------------------===//
// Main compilation entry points
//===----------------------------------------------------------------------===//

Error Driver::compileFromFile(StringRef InputFile, raw_ostream &OS,
                              const CompilationOptions &Options) {
    auto BufferOrError = MemoryBuffer::getFile(InputFile);
    if (!BufferOrError)
        return createFileError(InputFile, BufferOrError.getError());

    return compileFromBuffer(**BufferOrError, OS, Options, InputFile);
}

Error Driver::compileFromBuffer(const MemoryBuffer &InputBuffer,
                                raw_ostream &OS,
                                const CompilationOptions &Options,
                                StringRef BufferName) {
    return compileFromBufferWithConstexprs(InputBuffer, OS, Options, BufferName,
                                           /*ConstexprsJSON=*/"");
}

Error Driver::compileFromBufferWithConstexprs(const MemoryBuffer &InputBuffer,
                                              raw_ostream &OS,
                                              const CompilationOptions &Options,
                                              StringRef BufferName,
                                              StringRef ConstexprsJSON) {
    auto ASTOrError = parseBuffer(InputBuffer, BufferName);
    if (!ASTOrError)
        return ASTOrError.takeError();

    ast::ASTNode *AST = *ASTOrError;

    if (Options.Stage == OutputStage::AST)
        return emitAST(AST, OS);

    ir::MLIRGenerator Generator(Impl->IRContext.get(), Impl->DiagnosticManager);
    Generator.CreateModule();

    if (auto E = Generator.InjectConstexprs(ConstexprsJSON))
        return E;

    mlir::ModuleOp MLIRModule = Generator.Generate(AST);
    if (!MLIRModule)
        return createStringError(inconvertibleErrorCode(),
                                 "Failed to generate MLIR from AST");

    if (Options.Stage == OutputStage::MLIR)
        return emitMLIR(MLIRModule, OS);

    if (Options.Stage == OutputStage::LLVMIR) {
        auto LLVMModuleOrError = generateLLVMIR(MLIRModule, Options);
        if (!LLVMModuleOrError)
            return LLVMModuleOrError.takeError();
        return emitLLVMIR(**LLVMModuleOrError, OS);
    }

    if (Options.Stage == OutputStage::Binary) {
        auto BinaryOrError = generateBinary(MLIRModule, Options);
        if (!BinaryOrError)
            return BinaryOrError.takeError();
        return emitBinary(*BinaryOrError, OS);
    }

    return Error::success();
}

//===----------------------------------------------------------------------===//
// Individual compilation stages
//===----------------------------------------------------------------------===//

Expected<ast::ASTNode *> Driver::parseBuffer(const MemoryBuffer &Buffer,
                                             StringRef BufferName) {
    frontend::AveLangParser Parser(Impl->ASTContext, Impl->DiagnosticManager);
    Parser.ParseFromBuffer(Buffer, BufferName);

    ast::ASTNode *Result = Parser.GetModule();
    if (!Result)
        return createStringError(inconvertibleErrorCode(),
                                 "Failed to parse input buffer");

    return Result;
}

Expected<mlir::ModuleOp> Driver::generateMLIR(ast::ASTNode *AST) {
    if (!AST)
        return createStringError(inconvertibleErrorCode(), "AST is null");

    ir::MLIRGenerator Generator(Impl->IRContext.get(), Impl->DiagnosticManager);
    mlir::ModuleOp Result = Generator.Generate(AST);

    if (!Result)
        return createStringError(inconvertibleErrorCode(),
                                 "Failed to generate MLIR from AST");

    return Result;
}

Expected<std::unique_ptr<::llvm::Module>>
Driver::generateLLVMIR(mlir::ModuleOp MLIRModule,
                       const CompilationOptions &Options) {
    if (!MLIRModule)
        return createStringError(inconvertibleErrorCode(),
                                 "MLIR module is null");

    auto backendOrError = getBackendForTriple(Options.TargetTriple);
    if (!backendOrError)
        return backendOrError.takeError();

    (*backendOrError)->EnsureInitialized();

    target::gpu::LowerToLLVM Lowerer(Impl->IRContext.get());
    auto Result =
        Lowerer.compile(MLIRModule, *Impl->LLVMContext, toGPUOptions(Options));

    if (!Result)
        return createStringError(inconvertibleErrorCode(),
                                 "Failed to lower MLIR to LLVM IR");

    return std::move(Result);
}

Expected<std::string>
Driver::generateBinary(mlir::ModuleOp MLIRModule,
                       const CompilationOptions &Options) {
    if (!MLIRModule)
        return createStringError(inconvertibleErrorCode(),
                                 "MLIR module is null");

    auto LLVMModuleOrError = generateLLVMIR(MLIRModule, Options);
    if (!LLVMModuleOrError)
        return LLVMModuleOrError.takeError();

    auto backendOrError = getBackendForTriple(Options.TargetTriple);
    if (!backendOrError)
        return backendOrError.takeError();

    return (*backendOrError)
        ->generateBinary(**LLVMModuleOrError, toGPUOptions(Options));
}

//===----------------------------------------------------------------------===//
// Output helpers
//===----------------------------------------------------------------------===//

Error Driver::emitAST(ast::ASTNode *AST, raw_ostream &OS) {
    if (!AST)
        return createStringError(inconvertibleErrorCode(), "AST is null");

    // TODO: Implement AST serialization
    return createStringError(inconvertibleErrorCode(),
                             "AST emission not implemented yet");
}

Error Driver::emitMLIR(mlir::ModuleOp MLIRModule, raw_ostream &OS) {
    if (!MLIRModule)
        return createStringError(inconvertibleErrorCode(),
                                 "MLIR module is null");

    MLIRModule.print(OS);
    return Error::success();
}

Error Driver::emitLLVMIR(::llvm::Module &LLVMModule, raw_ostream &OS) {
    LLVMModule.print(OS, nullptr);
    return Error::success();
}

Error Driver::emitBinary(StringRef BinaryCode, raw_ostream &OS) {
    OS << BinaryCode;
    return Error::success();
}

} // namespace causalflow::avelang::driver
