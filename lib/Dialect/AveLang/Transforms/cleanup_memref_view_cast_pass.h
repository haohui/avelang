#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::dialect {

/// Clean up trivial byte-buffer view/cast patterns that inhibit later scalar
/// reasoning and promotion passes.
std::unique_ptr<mlir::Pass> createCleanupMemRefViewCastPass();

} // namespace causalflow::avelang::dialect

