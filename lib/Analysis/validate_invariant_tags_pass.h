#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::analysis {

std::unique_ptr<mlir::Pass> createValidateInvariantTagsPass();

} // namespace causalflow::avelang::analysis
