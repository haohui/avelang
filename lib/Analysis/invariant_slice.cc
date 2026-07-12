#include "invariant_slice.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <deque>

namespace causalflow::avelang::analysis {

namespace {

static bool belongsToFunction(mlir::Operation *op,
                              mlir::FunctionOpInterface func) {
    for (mlir::Operation *parent = op; parent; parent = parent->getParentOp())
        if (parent == func.getOperation()) return true;
    return false;
}

static bool isSeed(mlir::Operation *op) {
    // Tag operations and memory effects are the semantic boundaries of the
    // analysis. Calls are conservatively included because target intrinsics
    // commonly carry memory effects without a dialect-specific op interface.
    return mlir::isa<causalflow::avelang::dialect::TagBindOp,
                     causalflow::avelang::dialect::TagAssertEqOp,
                     causalflow::avelang::dialect::TagResetOp,
                     mlir::memref::LoadOp, mlir::memref::StoreOp,
                     mlir::func::CallOp, mlir::MemoryEffectOpInterface>(op);
}

} // namespace

InvariantSlice collectInvariantSlice(mlir::FunctionOpInterface func) {
    InvariantSlice slice;
    std::deque<mlir::Operation *> worklist;
    auto activate = [&](mlir::Operation *op) {
        if (op && belongsToFunction(op, func) &&
            slice.activeOps.insert(op).second)
            worklist.push_back(op);
    };

    func.getOperation()->walk([&](mlir::Operation *op) {
        if (auto nested = mlir::dyn_cast<mlir::FunctionOpInterface>(op);
            nested && nested != func)
            return mlir::WalkResult::skip();
        if (isSeed(op)) activate(op);
        return mlir::WalkResult::advance();
    });

    // A data-flow slice includes both directions: definitions explain an
    // active value's provenance, while uses retain transformations and memory
    // effects performed on it.
    while (!worklist.empty()) {
        mlir::Operation *op = worklist.front();
        worklist.pop_front();
        for (mlir::Value operand : op->getOperands())
            activate(operand.getDefiningOp());
        for (mlir::Value result : op->getResults())
            for (mlir::Operation *user : result.getUsers()) activate(user);
    }

    // MLIR SSA definitions precede their users in a block. Filtering lexical
    // order thus gives a deterministic topological order without reordering
    // side-effecting loads and stores.
    func.getOperation()->walk([&](mlir::Operation *op) {
        if (auto nested = mlir::dyn_cast<mlir::FunctionOpInterface>(op);
            nested && nested != func)
            return mlir::WalkResult::skip();
        if (slice.activeOps.contains(op)) slice.orderedOps.push_back(op);
        return mlir::WalkResult::advance();
    });
    return slice;
}

} // namespace causalflow::avelang::analysis
