#pragma once

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

#include <optional>

namespace causalflow::avelang::dialect {

inline std::optional<int64_t> getElementByteSize(mlir::Type elementType) {
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(elementType)) {
        auto scalarType = vectorType.getElementType();
        if (scalarType.isIndex()) {
            return 8 * vectorType.getNumElements();
        }
        if (!scalarType.isIntOrFloat()) {
            return std::nullopt;
        }
        int64_t bitWidth = scalarType.getIntOrFloatBitWidth();
        int64_t elementBytes = (bitWidth + 7) / 8;
        if (elementBytes <= 0) {
            return std::nullopt;
        }
        return elementBytes * vectorType.getNumElements();
    }

    if (elementType.isIndex()) {
        return 8;
    }

    if (!elementType.isIntOrFloat()) {
        return std::nullopt;
    }

    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    int64_t elementBytes = (bitWidth + 7) / 8;
    if (elementBytes <= 0) {
        return std::nullopt;
    }
    return elementBytes;
}

inline std::optional<int64_t> getStaticElementCount(mlir::MemRefType type) {
    int64_t count = 1;
    for (auto dim : type.getShape()) {
        if (mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return std::nullopt;
        }
        count *= dim;
    }
    return count;
}

inline std::optional<int64_t> getStaticByteSize(mlir::MemRefType type) {
    auto elemBytes = getElementByteSize(type.getElementType());
    auto elemCount = getStaticElementCount(type);
    if (!elemBytes || !elemCount) {
        return std::nullopt;
    }
    return (*elemBytes) * (*elemCount);
}

} // namespace causalflow::avelang::dialect

