#include "validate_invariant_tags_pass.h"
#include "Dialect/AveLang/IR/AveLangOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::analysis {

using namespace causalflow::avelang::dialect;

namespace {

class ValidateInvariantTagsPass
    : public mlir::PassWrapper<ValidateInvariantTagsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidateInvariantTagsPass)

    llvm::StringRef getArgument() const final {
        return "validate-invariant-tags";
    }

    llvm::StringRef getDescription() const final {
        return "Validate and materialize avelang invariant tag bindings";
    }

    void runOnOperation() final {
        auto func = getOperation();
        llvm::DenseSet<mlir::Value> seenBindings;
        llvm::SmallVector<mlir::Operation *> eraseOps;
        bool hadError = false;

        func.walk([&](TagBindOp op) {
            if (!seenBindings.insert(op.getTarget()).second) {
                op.emitOpError("duplicate tag binding for the same value");
                hadError = true;
                return;
            }

            if (op.getCaptures().size() != op.getTagCaptureNames().size()) {
                op.emitOpError("capture values must match tag capture names");
                hadError = true;
                return;
            }

            // Captured SSA values cannot be represented in a persistent MLIR
            // attribute.  Keep these bindings until late validation, where
            // their operands are translated into solver terms.
            if (!op.getCaptures().empty()) return;

            mlir::Builder builder(op.getContext());
            llvm::SmallVector<mlir::NamedAttribute> attrs;
            auto name = op.getTagNameAttr();
            if (name && !name.getValue().empty()) {
                attrs.push_back(builder.getNamedAttr("name", name));
            }
            attrs.push_back(builder.getNamedAttr("inputs", op.getTagInputs()));
            attrs.push_back(builder.getNamedAttr("exprs", op.getTagExprs()));

            auto dict = mlir::DictionaryAttr::get(op.getContext(), attrs);

            if (auto blockArg =
                    mlir::dyn_cast<mlir::BlockArgument>(op.getTarget())) {
                if (blockArg.getOwner() != &func.getFunctionBody().front()) {
                    op.emitOpError(
                        "only entry-block arguments may be tag-bound");
                    hadError = true;
                    return;
                }
                func.setArgAttr(blockArg.getArgNumber(),
                                "avelang.invariant.tag", dict);
            } else if (auto result =
                           mlir::dyn_cast<mlir::OpResult>(op.getTarget())) {
                auto attrName = llvm::formatv("avelang.invariant.tag.{0}",
                                              result.getResultNumber())
                                    .str();
                result.getOwner()->setAttr(attrName, dict);
            }

            eraseOps.push_back(op);
        });

        func.walk([&](TagAssertEqOp op) {
            if (mlir::isa<MemRefType>(op.getLhs().getType()) ||
                mlir::isa<MemRefType>(op.getRhs().getType())) {
                op.emitOpError(
                    "tag assertions must compare values, not memrefs");
                hadError = true;
                return;
            }
        });

        func.walk([&](TagResetOp) {});

        if (hadError) {
            signalPassFailure();
            return;
        }

        for (auto *op : eraseOps) {
            op->erase();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createValidateInvariantTagsPass() {
    return std::make_unique<ValidateInvariantTagsPass>();
}

} // namespace causalflow::avelang::analysis
