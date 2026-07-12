#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <string>

namespace causalflow::avelang::analysis {

enum class Z3Result {
    Sat,
    Unsat,
    Unknown,
};

struct Z3Response {
    Z3Result result;
    std::string stdout_text;
    std::string stderr_text;
};

llvm::Expected<Z3Response> runZ3SMTLIB(llvm::StringRef smtlib);

} // namespace causalflow::avelang::analysis
