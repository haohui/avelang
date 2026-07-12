#include "invariant_z3.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::analysis {

namespace {

constexpr llvm::StringLiteral kDefaultZ3Path(AVELANG_Z3_EXECUTABLE);

} // namespace

llvm::Expected<Z3Response> runZ3SMTLIB(llvm::StringRef smtlib) {
    auto inputTemp = llvm::sys::fs::TempFile::create("/tmp/avelang-z3-%%%%%%.smt2");
    if (!inputTemp) {
        return llvm::joinErrors(
            llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "failed to create temporary z3 input"),
            inputTemp.takeError());
    }
    auto outputTemp =
        llvm::sys::fs::TempFile::create("/tmp/avelang-z3-%%%%%%.out");
    if (!outputTemp) {
        llvm::consumeError(inputTemp->discard());
        return llvm::joinErrors(
            llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "failed to create temporary z3 output"),
            outputTemp.takeError());
    }

    {
        llvm::raw_fd_ostream os(inputTemp->FD, /*shouldClose=*/false);
        os << smtlib;
        os.flush();
    }

    std::vector<llvm::StringRef> args = {
        kDefaultZ3Path,
        "-smt2",
        inputTemp->TmpName,
    };
    std::optional<llvm::StringRef> redirects[] = {
        std::nullopt,
        outputTemp->TmpName,
        outputTemp->TmpName,
    };

    std::string errMsg;
    int result = llvm::sys::ExecuteAndWait(kDefaultZ3Path, args,
                                           /*Env=*/std::nullopt,
                                           redirects,
                                           /*SecondsToWait=*/0,
                                           /*MemoryLimit=*/0, &errMsg);

    auto readOutput = llvm::MemoryBuffer::getFile(outputTemp->TmpName);
    std::string outputText;
    if (readOutput) {
        outputText = (*readOutput)->getBuffer().str();
    }

    llvm::consumeError(inputTemp->discard());
    llvm::consumeError(outputTemp->discard());

    if (result != 0) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "z3 invocation failed: " + (errMsg.empty() ? outputText : errMsg));
    }

    llvm::StringRef trimmed = llvm::StringRef(outputText).trim();
    auto firstLine = trimmed.split('\n').first;
    Z3Result z3Result = Z3Result::Unknown;
    if (firstLine == "sat") {
        z3Result = Z3Result::Sat;
    } else if (firstLine == "unsat") {
        z3Result = Z3Result::Unsat;
    } else if (firstLine == "unknown") {
        z3Result = Z3Result::Unknown;
    } else {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "unexpected z3 output: " + firstLine.str());
    }

    return Z3Response{
        z3Result,
        outputText,
        errMsg,
    };
}

} // namespace causalflow::avelang::analysis
