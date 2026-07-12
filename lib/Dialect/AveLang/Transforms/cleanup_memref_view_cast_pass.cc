#include "cleanup_memref_view_cast_pass.h"
#include "memref_byte_size_utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>

namespace causalflow::avelang::dialect {

namespace {

bool isPrivateMemorySpace(mlir::Attribute memorySpace) {
    if (auto integerSpace = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
            memorySpace)) {
        return integerSpace.getInt() == 5;
    }

    auto gpuSpace = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
        memorySpace);
    return gpuSpace &&
           gpuSpace.getValue() == mlir::gpu::AddressSpace::Private;
}
class RewritePrivateSingleElementByteViewPattern
    : public mlir::OpRewritePattern<mlir::memref::ViewOp> {
  public:
    using OpRewritePattern<mlir::memref::ViewOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::memref::ViewOp op,
                    mlir::PatternRewriter &rewriter) const final {
        auto resultType = mlir::dyn_cast<mlir::MemRefType>(op.getType());
        if (!resultType || !resultType.hasStaticShape() ||
            resultType.getNumElements() != 1 || !resultType.getLayout().isIdentity() ||
            !isPrivateMemorySpace(resultType.getMemorySpace())) {
            return mlir::failure();
        }

        if (!op.getSizes().empty() || !mlir::matchPattern(op.getByteShift(), mlir::m_Zero())) {
            return mlir::failure();
        }

        auto sourceType = mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
        auto sourceAlloca = op.getSource().getDefiningOp<mlir::memref::AllocaOp>();
        if (!sourceType || !sourceAlloca || !sourceType.hasStaticShape() ||
            sourceType.getRank() != 1 || sourceType.getElementType() != rewriter.getI8Type() ||
            !sourceType.getLayout().isIdentity() || !sourceAlloca->hasOneUse()) {
            return mlir::failure();
        }

        if (sourceType.getMemorySpace() != resultType.getMemorySpace()) {
            return mlir::failure();
        }

        auto sourceBytes = getStaticByteSize(sourceType);
        auto resultBytes = getStaticByteSize(resultType);
        if (!sourceBytes || !resultBytes || *sourceBytes != *resultBytes) {
            return mlir::failure();
        }

        auto typedAlloca = rewriter.create<mlir::memref::AllocaOp>(
            op.getLoc(), resultType, mlir::ValueRange{},
            sourceAlloca.getAlignmentAttr());
        rewriter.replaceOp(op, typedAlloca.getResult());
        rewriter.eraseOp(sourceAlloca);
        return mlir::success();
    }
};

class CleanupMemRefViewCastPass
    : public mlir::PassWrapper<CleanupMemRefViewCastPass,
                               mlir::OperationPass<>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CleanupMemRefViewCastPass)

    llvm::StringRef getArgument() const final {
        return "cleanup-memref-view-cast";
    }

    llvm::StringRef getDescription() const final {
        return "Clean up trivial memref.view/cast patterns";
    }

    void runOnOperation() final {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<RewritePrivateSingleElementByteViewPattern>(
            &getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCleanupMemRefViewCastPass() {
    return std::make_unique<CleanupMemRefViewCastPass>();
}

} // namespace causalflow::avelang::dialect

