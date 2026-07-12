#pragma once

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

namespace causalflow::avelang::analysis {

// The conservative, bidirectional program slice consumed by invariant
// validation. `orderedOps` preserves lexical SSA order and therefore also the
// ordering of side-effecting memory operations.
struct InvariantSlice {
    llvm::DenseSet<mlir::Operation *> activeOps;
    llvm::SmallVector<mlir::Operation *> orderedOps;
};

InvariantSlice collectInvariantSlice(mlir::FunctionOpInterface func);

} // namespace causalflow::avelang::analysis
