#include "late_validate_invariant_tags_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "invariant_slice.h"
#include "invariant_z3.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <cctype>
#include <memory>
#include <optional>
#include <string>

namespace causalflow::avelang::analysis {

using namespace causalflow::avelang::dialect;

namespace {

struct SymbolicExpr;
using SymbolicExprPtr = std::shared_ptr<SymbolicExpr>;

struct SymbolicExpr {
    enum class Kind {
        Constant,
        Symbol,
        Add,
        Sub,
        Mul,
        Div,
        Rem,
    };

    Kind kind;
    int64_t constant = 0;
    std::string symbol;
    SymbolicExprPtr lhs;
    SymbolicExprPtr rhs;
};

struct TagValue {
    bool known = false;
    llvm::SmallVector<SymbolicExprPtr> components;
};

enum class ByteTagKind { Bottom, Concrete, Top };

// The native analysis domain. Every byte is independently bottom, concrete,
// or top; typed values are merely ordered byte ranges.
struct ByteTag {
    ByteTagKind kind = ByteTagKind::Bottom;
    TagValue concrete;
    // A raw access carries the descriptor-range condition under which its
    // provenance exists.  The condition is proved together with every tag
    // assertion that consumes the byte; an out-of-range read is bottom.
    std::string validityCondition;
};

struct TagTemplate {
    llvm::SmallVector<std::string> inputs;
    llvm::StringMap<SymbolicExprPtr> captures;
    llvm::SmallVector<std::string> exprs;
};

struct ValidationState {
    llvm::DenseMap<mlir::Value, std::string> symbolNames;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<ByteTag>> valueTags;
    llvm::StringMap<ByteTag> memoryTags;
    llvm::DenseMap<mlir::Value, TagTemplate> boundMemRefs;
    llvm::StringMap<llvm::SmallVector<std::string>> symbolConstraints;
    int64_t nextSymbolId = 0;
};

static bool tagsStructurallyEqual(const TagValue &lhs, const TagValue &rhs);

static SymbolicExprPtr makeConstant(int64_t value) {
    auto expr = std::make_shared<SymbolicExpr>();
    expr->kind = SymbolicExpr::Kind::Constant;
    expr->constant = value;
    return expr;
}

static SymbolicExprPtr makeSymbol(llvm::StringRef symbol) {
    auto expr = std::make_shared<SymbolicExpr>();
    expr->kind = SymbolicExpr::Kind::Symbol;
    expr->symbol = symbol.str();
    return expr;
}

static SymbolicExprPtr makeBinary(SymbolicExpr::Kind kind,
                                  SymbolicExprPtr lhs,
                                  SymbolicExprPtr rhs) {
    auto expr = std::make_shared<SymbolicExpr>();
    expr->kind = kind;
    expr->lhs = std::move(lhs);
    expr->rhs = std::move(rhs);
    return expr;
}

static std::string renderIntLiteral(int64_t value) {
    if (value < 0) {
        return llvm::formatv("(- {0})", -value).str();
    }
    return std::to_string(value);
}

static std::string exprText(const SymbolicExprPtr &expr) {
    switch (expr->kind) {
    case SymbolicExpr::Kind::Constant:
        return renderIntLiteral(expr->constant);
    case SymbolicExpr::Kind::Symbol:
        return expr->symbol;
    case SymbolicExpr::Kind::Add:
        return llvm::formatv("(+ {0} {1})", exprText(expr->lhs),
                             exprText(expr->rhs))
            .str();
    case SymbolicExpr::Kind::Sub:
        return llvm::formatv("(- {0} {1})", exprText(expr->lhs),
                             exprText(expr->rhs))
            .str();
    case SymbolicExpr::Kind::Mul:
        return llvm::formatv("(* {0} {1})", exprText(expr->lhs),
                             exprText(expr->rhs))
            .str();
    case SymbolicExpr::Kind::Div:
        return llvm::formatv("(div {0} {1})", exprText(expr->lhs),
                             exprText(expr->rhs))
            .str();
    case SymbolicExpr::Kind::Rem:
        return llvm::formatv("(mod {0} {1})", exprText(expr->lhs),
                             exprText(expr->rhs))
            .str();
    }
    llvm_unreachable("unknown expr kind");
}

static bool exprEquals(const SymbolicExprPtr &lhs, const SymbolicExprPtr &rhs) {
    return exprText(lhs) == exprText(rhs);
}

static void collectSymbols(const SymbolicExprPtr &expr,
                           llvm::StringSet<> &symbols) {
    switch (expr->kind) {
    case SymbolicExpr::Kind::Constant:
        return;
    case SymbolicExpr::Kind::Symbol:
        symbols.insert(expr->symbol);
        return;
    case SymbolicExpr::Kind::Add:
    case SymbolicExpr::Kind::Sub:
    case SymbolicExpr::Kind::Mul:
    case SymbolicExpr::Kind::Div:
    case SymbolicExpr::Kind::Rem:
        collectSymbols(expr->lhs, symbols);
        collectSymbols(expr->rhs, symbols);
        return;
    }
}

static std::optional<int64_t> getConstantInt(mlir::Value value) {
    if (auto indexOp = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return indexOp.value();
    }
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

static SymbolicExprPtr buildExpr(mlir::Value value, ValidationState &state) {
    if (auto constant = getConstantInt(value)) {
        return makeConstant(*constant);
    }

    if (auto castOp = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return buildExpr(castOp.getIn(), state);
    }
    if (auto addOp = value.getDefiningOp<mlir::arith::AddIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Add,
                          buildExpr(addOp.getLhs(), state),
                          buildExpr(addOp.getRhs(), state));
    }
    if (auto subOp = value.getDefiningOp<mlir::arith::SubIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Sub,
                          buildExpr(subOp.getLhs(), state),
                          buildExpr(subOp.getRhs(), state));
    }
    if (auto mulOp = value.getDefiningOp<mlir::arith::MulIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Mul,
                          buildExpr(mulOp.getLhs(), state),
                          buildExpr(mulOp.getRhs(), state));
    }
    if (auto divSOp = value.getDefiningOp<mlir::arith::DivSIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Div,
                          buildExpr(divSOp.getLhs(), state),
                          buildExpr(divSOp.getRhs(), state));
    }
    if (auto floorDivOp = value.getDefiningOp<mlir::arith::FloorDivSIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Div,
                          buildExpr(floorDivOp.getLhs(), state),
                          buildExpr(floorDivOp.getRhs(), state));
    }
    if (auto remSOp = value.getDefiningOp<mlir::arith::RemSIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Rem,
                          buildExpr(remSOp.getLhs(), state),
                          buildExpr(remSOp.getRhs(), state));
    }
    if (auto remUOp = value.getDefiningOp<mlir::arith::RemUIOp>()) {
        return makeBinary(SymbolicExpr::Kind::Rem,
                          buildExpr(remUOp.getLhs(), state),
                          buildExpr(remUOp.getRhs(), state));
    }

    auto it = state.symbolNames.find(value);
    if (it == state.symbolNames.end()) {
        auto symbol = llvm::formatv("v{0}", state.nextSymbolId++).str();
        it = state.symbolNames.try_emplace(value, symbol).first;

        // Preserve static scf.for bounds so Z3 can prove identities such as
        // (k * 4 + lane) div 4 == k for lane in [0, 4).
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(
                blockArg.getOwner()->getParentOp());
            if (forOp && blockArg.getArgNumber() == 0) {
                auto lower = buildExpr(forOp.getLowerBound(), state);
                auto upper = buildExpr(forOp.getUpperBound(), state);
                state.symbolConstraints[it->second].push_back(
                    "(>= " + it->second + " " + exprText(lower) + ")");
                state.symbolConstraints[it->second].push_back(
                    "(< " + it->second + " " + exprText(upper) + ")");
            }
        }
    }
    return makeSymbol(it->second);
}

static std::optional<TagTemplate> getTagTemplate(mlir::Value value) {
    if (!value) return std::nullopt;
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(
            blockArg.getOwner()->getParentOp());
        if (!func || blockArg.getOwner() != &func.getFunctionBody().front()) {
            return std::nullopt;
        }

        auto attr = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            func.getArgAttr(blockArg.getArgNumber(), "avelang.invariant.tag"));
        if (!attr) {
            return std::nullopt;
        }

        auto inputs =
            mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr.get("inputs"));
        auto exprs = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr.get("exprs"));
        if (!inputs || !exprs) {
            return std::nullopt;
        }

        TagTemplate result;
        for (auto attrValue : inputs) {
            auto str = mlir::dyn_cast<mlir::StringAttr>(attrValue);
            if (!str) {
                return std::nullopt;
            }
            result.inputs.push_back(str.str());
        }
        for (auto attrValue : exprs) {
            auto str = mlir::dyn_cast<mlir::StringAttr>(attrValue);
            if (!str) {
                return std::nullopt;
            }
            result.exprs.push_back(str.str());
        }
        return result;
    }

    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result) {
        return std::nullopt;
    }

    auto attrName =
        llvm::formatv("avelang.invariant.tag.{0}", result.getResultNumber()).str();
    auto attr = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
        result.getOwner()->getAttr(attrName));
    if (!attr) {
        return std::nullopt;
    }

    auto inputs = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr.get("inputs"));
    auto exprs = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr.get("exprs"));
    if (!inputs || !exprs) {
        return std::nullopt;
    }

    TagTemplate templateValue;
    for (auto attrValue : inputs) {
        auto str = mlir::dyn_cast<mlir::StringAttr>(attrValue);
        if (!str) {
            return std::nullopt;
        }
        templateValue.inputs.push_back(str.str());
    }
    for (auto attrValue : exprs) {
        auto str = mlir::dyn_cast<mlir::StringAttr>(attrValue);
        if (!str) {
            return std::nullopt;
        }
        templateValue.exprs.push_back(str.str());
    }
    return templateValue;
}

static std::optional<TagTemplate> getTagTemplate(TagBindOp bind,
                                                 ValidationState &state) {
    if (bind.getCaptures().size() != bind.getTagCaptureNames().size())
        return std::nullopt;

    TagTemplate result;
    for (auto attr : bind.getTagInputs()) {
        auto string = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!string) return std::nullopt;
        result.inputs.push_back(string.str());
    }
    for (auto [nameAttr, value] :
         llvm::zip(bind.getTagCaptureNames(), bind.getCaptures())) {
        auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
        if (!name) return std::nullopt;
        result.captures[name.getValue()] = buildExpr(value, state);
    }
    for (auto attr : bind.getTagExprs()) {
        auto string = mlir::dyn_cast<mlir::StringAttr>(attr);
        if (!string) return std::nullopt;
        result.exprs.push_back(string.str());
    }
    return result;
}

class TagExprParser {
  public:
    TagExprParser(llvm::StringRef input,
                  const llvm::StringMap<SymbolicExprPtr> &substitutions)
        : input_(input), substitutions_(substitutions) {}

    std::optional<SymbolicExprPtr> parse() {
        auto expr = parseExpr();
        if (!expr) {
            return std::nullopt;
        }
        skipWhitespace();
        if (pos_ != input_.size()) {
            return std::nullopt;
        }
        return expr;
    }

  private:
    void skipWhitespace() {
        while (pos_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    bool consume(char c) {
        skipWhitespace();
        if (pos_ >= input_.size() || input_[pos_] != c) {
            return false;
        }
        ++pos_;
        return true;
    }

    std::optional<llvm::StringRef> parseToken() {
        skipWhitespace();
        size_t start = pos_;
        while (pos_ < input_.size() &&
               !std::isspace(static_cast<unsigned char>(input_[pos_])) &&
               input_[pos_] != '(' && input_[pos_] != ')') {
            ++pos_;
        }
        if (start == pos_) {
            return std::nullopt;
        }
        return input_.slice(start, pos_);
    }

    std::optional<int64_t> parseIntegerLiteral() {
        skipWhitespace();
        size_t start = pos_;
        if (pos_ < input_.size() && input_[pos_] == '-') {
            ++pos_;
        }
        size_t digitsStart = pos_;
        while (pos_ < input_.size() &&
               std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
        if (digitsStart == pos_) {
            pos_ = start;
            return std::nullopt;
        }

        int64_t value = 0;
        if (!llvm::to_integer(input_.slice(start, pos_), value, 10)) {
            pos_ = start;
            return std::nullopt;
        }
        return value;
    }

    std::optional<SymbolicExprPtr> parseExpr() {
        skipWhitespace();
        if (consume('(')) {
            if (consume('-')) {
                auto operand = parseExpr();
                if (!operand || !consume(')')) {
                    return std::nullopt;
                }
                return makeBinary(SymbolicExpr::Kind::Sub, makeConstant(0),
                                  *operand);
            }

            auto op = parseToken();
            if (!op) {
                return std::nullopt;
            }

            auto lhs = parseExpr();
            auto rhs = parseExpr();
            if (!lhs || !rhs || !consume(')')) {
                return std::nullopt;
            }

            if (*op == "add") {
                return makeBinary(SymbolicExpr::Kind::Add, *lhs, *rhs);
            }
            if (*op == "sub") {
                return makeBinary(SymbolicExpr::Kind::Sub, *lhs, *rhs);
            }
            if (*op == "mult") {
                return makeBinary(SymbolicExpr::Kind::Mul, *lhs, *rhs);
            }
            if (*op == "floordiv") {
                return makeBinary(SymbolicExpr::Kind::Div, *lhs, *rhs);
            }
            if (*op == "mod") {
                return makeBinary(SymbolicExpr::Kind::Rem, *lhs, *rhs);
            }
            return std::nullopt;
        }

        if (auto integer = parseIntegerLiteral()) {
            return makeConstant(*integer);
        }

        auto token = parseToken();
        if (!token) {
            return std::nullopt;
        }

        auto it = substitutions_.find(*token);
        if (it == substitutions_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    llvm::StringRef input_;
    const llvm::StringMap<SymbolicExprPtr> &substitutions_;
    size_t pos_ = 0;
};

static TagValue instantiateTag(const TagTemplate &tagTemplate,
                               mlir::ValueRange indices,
                               ValidationState &state) {
    if (tagTemplate.inputs.size() != indices.size()) {
        return {};
    }

    llvm::StringMap<SymbolicExprPtr> substitutions;
    for (const auto &[name, value] : tagTemplate.captures)
        substitutions[name] = value;
    for (auto [input, index] : llvm::zip(tagTemplate.inputs, indices)) {
        substitutions[input] = buildExpr(index, state);
    }

    TagValue result;
    result.known = true;
    for (const auto &expr : tagTemplate.exprs) {
        TagExprParser parser(expr, substitutions);
        auto parsed = parser.parse();
        if (!parsed) {
            result.known = false;
            result.components.clear();
            return result;
        }
        result.components.push_back(*parsed);
    }
    return result;
}

static TagValue instantiateTag(const TagTemplate &tagTemplate,
                               llvm::ArrayRef<SymbolicExprPtr> indices) {
    if (tagTemplate.inputs.size() != indices.size()) return {};
    llvm::StringMap<SymbolicExprPtr> substitutions;
    for (const auto &[name, value] : tagTemplate.captures)
        substitutions[name] = value;
    for (auto [input, index] : llvm::zip(tagTemplate.inputs, indices))
        substitutions[input] = index;

    TagValue result;
    result.known = true;
    for (const auto &expr : tagTemplate.exprs) {
        TagExprParser parser(expr, substitutions);
        auto parsed = parser.parse();
        if (!parsed) return {};
        result.components.push_back(*parsed);
    }
    return result;
}

static std::optional<size_t> getByteWidth(mlir::Type type) {
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
        return (integer.getWidth() + 7) / 8;
    }
    if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
        return floating.getWidth() / 8;
    }
    if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
        if (vector.getRank() != 1) return std::nullopt;
        auto elementWidth = getByteWidth(vector.getElementType());
        if (!elementWidth) return std::nullopt;
        return vector.getNumElements() * *elementWidth;
    }
    return std::nullopt;
}

static std::optional<std::pair<unsigned, mlir::Type>>
getMemrefRankAndElementType(mlir::Value memref) {
    if (auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType()))
        return std::pair<unsigned, mlir::Type>{type.getRank(), type.getElementType()};
    if (auto type = mlir::dyn_cast<MemRefType>(memref.getType()))
        return std::pair<unsigned, mlir::Type>{type.getRank(), type.getElementType()};
    return std::nullopt;
}

// Raw buffer descriptors are deliberately not treated as opaque pointers by
// validation.  make_rsrc builds a vector<4xi32> by inserting the low pointer
// word, high pointer word, range, and configuration in that order.  Following
// the insert chain gives the validator the same base object that was tagged at
// the source level, without requiring a separate alias analysis.
static mlir::Value getStaticVectorLane(mlir::Value value, unsigned lane) {
    while (auto insert = value.getDefiningOp<mlir::vector::InsertOp>()) {
        auto position = insert.getStaticPosition();
        if (position.size() != 1) return {};
        if (position.front() == static_cast<int64_t>(lane))
            return insert.getValueToStore();
        value = insert.getDest();
    }
    return {};
}

static mlir::Value getMemrefFromPointer(mlir::Value value) {
    if (!value) return {};
    llvm::SmallVector<mlir::Value> worklist{value};
    llvm::DenseSet<mlir::Value> visited;
    while (!worklist.empty()) {
        auto current = worklist.pop_back_val();
        if (!visited.insert(current).second) continue;
        if (auto pointer = current.getDefiningOp<
                AveLangMemRefExtractAlignedPointerAsIndexOp>())
            return pointer.getSource();
        if (auto pointer = current.getDefiningOp<
                mlir::memref::ExtractAlignedPointerAsIndexOp>())
            return pointer.getSource();
        if (auto def = current.getDefiningOp())
            for (auto operand : def->getOperands()) worklist.push_back(operand);
    }
    return {};
}

static mlir::Value getRawBufferMemref(mlir::Value descriptor) {
    // Either pointer word reaches the same extract operation.  Using the low
    // word keeps this tolerant of the high-word shifts/casts used by make_rsrc.
    return getMemrefFromPointer(getStaticVectorLane(descriptor, 0));
}

static bool isRawBufferLoadOperation(mlir::Operation *op) {
    if (mlir::isa<AMDGPURawBufferLoadOp>(op)) return true;
    auto name = op->getName().getStringRef();
    return name.contains("raw.buffer.load") && !name.contains("load.lds");
}

static bool isRawBufferStoreOperation(mlir::Operation *op) {
    if (mlir::isa<AMDGPURawBufferStoreOp>(op)) return true;
    return op->getName().getStringRef().contains("raw.buffer.store");
}

static bool isRawBufferLoadLdsCall(mlir::Operation *op) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
    return call && call.getCallee() ==
                       "_avelang_amdgpu_llvm_amdgcn_raw_buffer_load_lds_u32";
}


static llvm::SmallVector<ByteTag> expandTagToBytes(const TagValue &tag,
                                                    mlir::Type type) {
    auto width = getByteWidth(type);
    if (!tag.known || !width) return {};
    llvm::SmallVector<ByteTag> result;
    result.assign(*width, ByteTag{ByteTagKind::Concrete, tag});
    return result;
}

static llvm::SmallVector<ByteTag> bottomBytes(mlir::Type type) {
    llvm::SmallVector<ByteTag> result;
    if (auto width = getByteWidth(type)) result.assign(*width, ByteTag{});
    return result;
}

// The join operation from the paper. Bottom is the identity and incompatible
// concrete provenance conservatively becomes top.
static ByteTag mergeByteTag(const ByteTag &lhs, const ByteTag &rhs) {
    if (lhs.kind == ByteTagKind::Bottom) return rhs;
    if (rhs.kind == ByteTagKind::Bottom) return lhs;
    if (lhs.kind == ByteTagKind::Top || rhs.kind == ByteTagKind::Top)
        return ByteTag{ByteTagKind::Top, {}};
    if (tagsStructurallyEqual(lhs.concrete, rhs.concrete) &&
        lhs.validityCondition == rhs.validityCondition)
        return lhs;
    return ByteTag{ByteTagKind::Top, {}};
}

static llvm::SmallVector<ByteTag> valueBytes(mlir::Value value,
                                              ValidationState &state) {
    if (auto it = state.valueTags.find(value); it != state.valueTags.end())
        return it->second;
    return bottomBytes(value.getType());
}

struct MemrefAccess {
    mlir::Value root;
    llvm::SmallVector<SymbolicExprPtr> indices;
};

static std::optional<SymbolicExprPtr>
buildExpr(mlir::OpFoldResult value, ValidationState &state) {
    if (auto dynamic = llvm::dyn_cast_if_present<mlir::Value>(value))
        return buildExpr(dynamic, state);
    auto attribute = llvm::dyn_cast_if_present<mlir::Attribute>(value);
    auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute);
    if (!attr) return std::nullopt;
    return makeConstant(attr.getInt());
}

// Resolve a lowered memref access back to the tagged root.  Subviews and
// reinterpret casts carry affine offsets/strides, so their result attributes
// are not independent tags: each result index is translated before template
// instantiation.  This is the missing link from v_view back to v_flat.
static std::optional<MemrefAccess>
resolveMemrefAccessExpr(mlir::Value memref,
                        llvm::SmallVector<SymbolicExprPtr> indexExprs,
                        ValidationState &state) {

    if (auto subview = memref.getDefiningOp<mlir::memref::SubViewOp>()) {
        auto offsets = subview.getMixedOffsets();
        auto strides = subview.getMixedStrides();
        if (offsets.size() != indexExprs.size() ||
            strides.size() != indexExprs.size())
            return std::nullopt;
        llvm::SmallVector<SymbolicExprPtr> sourceIndices;
        sourceIndices.reserve(indexExprs.size());
        for (auto [index, offset, stride] :
             llvm::zip(indexExprs, offsets, strides)) {
            auto offsetExpr = buildExpr(offset, state);
            auto strideExpr = buildExpr(stride, state);
            if (!offsetExpr || !strideExpr) return std::nullopt;
            sourceIndices.push_back(makeBinary(
                SymbolicExpr::Kind::Add, *offsetExpr,
                makeBinary(SymbolicExpr::Kind::Mul, index, *strideExpr)));
        }
        auto source = subview.getSource();
        return resolveMemrefAccessExpr(source, std::move(sourceIndices), state);
    }

    if (auto reinterpret = memref.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
        auto offsets = reinterpret.getMixedOffsets();
        auto strides = reinterpret.getMixedStrides();
        if (offsets.size() != indexExprs.size() ||
            strides.size() != indexExprs.size())
            return std::nullopt;
        llvm::SmallVector<SymbolicExprPtr> sourceIndices;
        sourceIndices.reserve(indexExprs.size());
        for (auto [index, offset, stride] :
             llvm::zip(indexExprs, offsets, strides)) {
            auto offsetExpr = buildExpr(offset, state);
            auto strideExpr = buildExpr(stride, state);
            if (!offsetExpr || !strideExpr) return std::nullopt;
            sourceIndices.push_back(makeBinary(
                SymbolicExpr::Kind::Add, *offsetExpr,
                makeBinary(SymbolicExpr::Kind::Mul, index, *strideExpr)));
        }
        auto source = reinterpret.getSource();
        return resolveMemrefAccessExpr(source, std::move(sourceIndices), state);
    }

    if (auto cast = memref.getDefiningOp<mlir::memref::CastOp>())
        return resolveMemrefAccessExpr(cast.getSource(), std::move(indexExprs), state);

    // A tagged view is the semantic root for its template (for example
    // v_flat).  Subviews/reinterpret casts above this point have already
    // translated their access into this root's logical coordinates.
    if (getTagTemplate(memref)) return MemrefAccess{memref, std::move(indexExprs)};

    if (auto view = memref.getDefiningOp<mlir::memref::ViewOp>()) {
        auto viewType = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
        auto sourceType =
            mlir::dyn_cast<mlir::MemRefType>(view.getSource().getType());
        llvm::SmallVector<int64_t> strides;
        int64_t offset = 0;
        if (!viewType || !sourceType || !sourceType.getElementType().isInteger(8) ||
            mlir::failed(viewType.getStridesAndOffset(strides, offset)) ||
            mlir::ShapedType::isDynamic(offset) ||
            strides.size() != indexExprs.size())
            return std::nullopt;
        auto elementBytes = getByteWidth(viewType.getElementType());
        if (!elementBytes) return std::nullopt;
        auto byteOffset = buildExpr(view.getByteShift(), state);
        if (!byteOffset) return std::nullopt;
        auto byteAddress = byteOffset;
        for (auto [index, stride] : llvm::zip(indexExprs, strides)) {
            if (mlir::ShapedType::isDynamic(stride)) return std::nullopt;
            byteAddress = makeBinary(
                SymbolicExpr::Kind::Add, byteAddress,
                makeBinary(SymbolicExpr::Kind::Mul, index,
                           makeConstant(stride * static_cast<int64_t>(*elementBytes))));
        }
        return resolveMemrefAccessExpr(view.getSource(), {byteAddress}, state);
    }

    // Untagged views continue to their backing storage so typed stores through
    // one view are visible through another.
    return MemrefAccess{memref, std::move(indexExprs)};
}

static std::optional<MemrefAccess>
resolveMemrefAccess(mlir::Value memref, mlir::ValueRange indices,
                    ValidationState &state) {
    llvm::SmallVector<SymbolicExprPtr> indexExprs;
    indexExprs.reserve(indices.size());
    for (auto index : indices) indexExprs.push_back(buildExpr(index, state));
    return resolveMemrefAccessExpr(memref, std::move(indexExprs), state);
}

static std::string memoryKey(mlir::Value base, mlir::ValueRange indices,
                             ValidationState &state) {
    if (auto access = resolveMemrefAccess(base, indices, state)) {
        std::string key = llvm::formatv(
            "p{0}", reinterpret_cast<uintptr_t>(access->root.getAsOpaquePointer())).str();
        for (const auto &index : access->indices) {
            key += "|";
            key += exprText(index);
        }
        return key;
    }
    std::string key =
        llvm::formatv("p{0}", reinterpret_cast<uintptr_t>(base.getAsOpaquePointer()))
            .str();
    for (mlir::Value index : indices) {
        key += "|";
        key += exprText(buildExpr(index, state));
    }
    return key;
}

static std::optional<llvm::SmallVector<ByteTag>>
buildTagForLoad(mlir::Value memref, mlir::ValueRange indices,
                mlir::Type resultType, ValidationState &state) {
    auto width = getByteWidth(resultType);
    if (!width) return std::nullopt;
    const auto baseKey = memoryKey(memref, indices, state);
    llvm::SmallVector<ByteTag> stored;
    bool hasStoredByte = false;
    for (size_t byte = 0; byte < *width; ++byte) {
        auto it = state.memoryTags.find(baseKey + ":" + std::to_string(byte));
        if (it == state.memoryTags.end()) break;
        stored.push_back(it->second);
        hasStoredByte = true;
    }
    if (hasStoredByte && stored.size() == *width) return stored;

    auto access = resolveMemrefAccess(memref, indices, state);
    auto tagMemref = access ? access->root : memref;

    auto boundIt = state.boundMemRefs.find(tagMemref);
    if (boundIt != state.boundMemRefs.end()) {
        auto instantiated = access ? instantiateTag(boundIt->second, access->indices)
                                   : instantiateTag(boundIt->second, indices, state);
        if (instantiated.known) {
            return expandTagToBytes(instantiated, resultType);
        }
        return std::nullopt;
    }

    if (auto maybeTemplate = getTagTemplate(tagMemref)) {
        state.boundMemRefs[tagMemref] = *maybeTemplate;
        auto instantiated = access ? instantiateTag(*maybeTemplate, access->indices)
                                   : instantiateTag(*maybeTemplate, indices, state);
        if (instantiated.known) {
            return expandTagToBytes(instantiated, resultType);
        }
    }

    // An untagged typed location is represented natively as bottom bytes.
    return bottomBytes(resultType);
}

// Model the raw-buffer address in bytes, then instantiate the source tag with
// the corresponding element index.  This intentionally follows the resource
// value rather than relying on operation ordering or alias analysis.
static llvm::SmallVector<ByteTag>
buildTagForRawBufferLoad(mlir::Value descriptor, mlir::Value vindex,
                         mlir::Value soffset, mlir::Type resultType,
                         ValidationState &state) {
    auto width = getByteWidth(resultType);
    auto memref = getRawBufferMemref(descriptor);
    if (!width || !memref) return bottomBytes(resultType);

    auto memrefInfo = getMemrefRankAndElementType(memref);
    if (!memrefInfo || memrefInfo->first != 1) return bottomBytes(resultType);
    auto elementBytes = getByteWidth(memrefInfo->second);
    if (!elementBytes || *elementBytes == 0) return bottomBytes(resultType);

    auto tagTemplate = getTagTemplate(memref);
    if (!tagTemplate || tagTemplate->inputs.size() != 1)
        return bottomBytes(resultType);

    auto offset = makeBinary(SymbolicExpr::Kind::Add, buildExpr(vindex, state),
                             buildExpr(soffset, state));
    auto rangeLane = getStaticVectorLane(descriptor, 2);
    if (!rangeLane) return bottomBytes(resultType);
    // AMD raw-buffer ranges are byte ranges.  The guard is deliberately kept
    // with the produced tag rather than asserted globally: an OOB load is
    // valid hardware behavior, but it cannot establish non-bottom provenance.
    auto inRange = llvm::formatv("(<= (+ {0} {1}) {2})", exprText(offset),
                                 *width,
                                 exprText(buildExpr(rangeLane, state))).str();
    llvm::SmallVector<ByteTag> result;
    result.reserve(*width);
    for (size_t byte = 0; byte < *width; ++byte) {
        auto byteAddress = makeBinary(SymbolicExpr::Kind::Add, offset,
                                      makeConstant(static_cast<int64_t>(byte)));
        auto elementIndex = makeBinary(SymbolicExpr::Kind::Div, byteAddress,
                                       makeConstant(*elementBytes));
        auto access = resolveMemrefAccessExpr(memref, {elementIndex}, state);
        auto tag = access ? instantiateTag(*tagTemplate, access->indices)
                          : instantiateTag(*tagTemplate, {elementIndex});
        result.push_back(tag.known ? ByteTag{ByteTagKind::Concrete, std::move(tag), inRange}
                                   : ByteTag{});
    }
    return result;
}

static std::optional<std::string>
rawBufferMemoryKey(mlir::Value descriptor, mlir::Value vindex,
                   mlir::Value soffset, ValidationState &state) {
    auto memref = getRawBufferMemref(descriptor);
    if (!memref) return std::nullopt;
    auto memrefInfo = getMemrefRankAndElementType(memref);
    if (!memrefInfo || memrefInfo->first != 1) return std::nullopt;
    auto elementBytes = getByteWidth(memrefInfo->second);
    if (!elementBytes || *elementBytes == 0) return std::nullopt;
    auto byteOffset = makeBinary(SymbolicExpr::Kind::Add, buildExpr(vindex, state),
                                 buildExpr(soffset, state));
    auto elementIndex = makeBinary(SymbolicExpr::Kind::Div, byteOffset,
                                   makeConstant(*elementBytes));
    auto access = resolveMemrefAccessExpr(memref, {elementIndex}, state);
    auto root = access ? access->root : memref;
    auto index = access ? access->indices.front() : elementIndex;
    return llvm::formatv("p{0}|{1}",
                         reinterpret_cast<uintptr_t>(root.getAsOpaquePointer()),
                         exprText(index)).str();
}

static std::optional<std::string>
pointerMemoryKey(mlir::Value pointer, ValidationState &state) {
    auto add = pointer.getDefiningOp<mlir::arith::AddIOp>();
    if (!add) return std::nullopt;
    auto memref = getMemrefFromPointer(add.getLhs());
    if (!memref) memref = getMemrefFromPointer(add.getRhs());
    if (!memref) return std::nullopt;
    auto memrefInfo = getMemrefRankAndElementType(memref);
    if (!memrefInfo || memrefInfo->first != 1) return std::nullopt;
    auto elementBytes = getByteWidth(memrefInfo->second);
    if (!elementBytes || *elementBytes == 0) return std::nullopt;
    auto offset = getMemrefFromPointer(add.getLhs()) ? add.getRhs() : add.getLhs();
    auto index = makeBinary(SymbolicExpr::Kind::Div, buildExpr(offset, state),
                            makeConstant(*elementBytes));
    return llvm::formatv("p{0}|{1}",
                         reinterpret_cast<uintptr_t>(memref.getAsOpaquePointer()),
                         exprText(index)).str();
}

static std::optional<std::string> buildEqualityQuery(
    llvm::ArrayRef<std::pair<TagValue, TagValue>> bytePairs,
    llvm::ArrayRef<std::string> validityConditions,
    const ValidationState &state) {
    if (bytePairs.empty() && validityConditions.empty()) return std::nullopt;

    llvm::StringSet<> symbols;
    for (const auto &[lhs, rhs] : bytePairs) {
        if (!lhs.known || !rhs.known ||
            lhs.components.size() != rhs.components.size()) {
            return std::nullopt;
        }
        for (auto [lhsExpr, rhsExpr] : llvm::zip(lhs.components, rhs.components)) {
            collectSymbols(lhsExpr, symbols);
            collectSymbols(rhsExpr, symbols);
        }
    }
    // Validity conditions for raw accesses may mention the descriptor range
    // even when the tag expression itself does not.  Declaring every symbol
    // allocated by the active program keeps those guards well-formed.
    for (const auto &[value, symbol] : state.symbolNames) {
        (void)value;
        symbols.insert(symbol);
    }

    std::string smt;
    llvm::raw_string_ostream os(smt);
    os << "(set-logic QF_NIA)\n";
    for (const auto &symbol : symbols.keys()) {
        os << "(declare-const " << symbol << " Int)\n";
        if (auto constraints = state.symbolConstraints.find(symbol);
            constraints != state.symbolConstraints.end()) {
            for (const auto &constraint : constraints->second) {
                os << "(assert " << constraint << ")\n";
            }
        }
    }
    os << "(assert (not (and";
    for (const auto &condition : validityConditions)
        os << " " << condition;
    for (const auto &[lhs, rhs] : bytePairs) {
        for (auto [lhsExpr, rhsExpr] : llvm::zip(lhs.components, rhs.components)) {
            os << " (= " << exprText(lhsExpr) << " " << exprText(rhsExpr) << ")";
        }
    }
    os << ")))\n(check-sat)\n";
    os.flush();
    return smt;
}

static bool tagsStructurallyEqual(const TagValue &lhs, const TagValue &rhs) {
    if (!lhs.known || !rhs.known || lhs.components.size() != rhs.components.size()) {
        return false;
    }
    for (auto [lhsExpr, rhsExpr] : llvm::zip(lhs.components, rhs.components)) {
        if (!exprEquals(lhsExpr, rhsExpr)) {
            return false;
        }
    }
    return true;
}

struct PendingTagAssertion {
    TagAssertEqOp op;
    llvm::SmallVector<std::pair<TagValue, TagValue>> bytePairs;
    llvm::SmallVector<std::string> validityConditions;
};

static mlir::LogicalResult validateFunction(mlir::FunctionOpInterface func) {
    ValidationState state;
    const auto slice = collectInvariantSlice(func);
    bool hadError = false;
    llvm::SmallVector<mlir::Operation *> eraseOps;
    llvm::SmallVector<PendingTagAssertion> pendingAssertions;

    // Interpret the byte-level transfer relation in topological program order.
    // `ordered` contains precisely the slice, so inactive instructions cannot
    // accidentally introduce state or solver symbols.
    for (mlir::Operation *op : slice.orderedOps) {

        if (auto bind = mlir::dyn_cast<TagBindOp>(op)) {
            auto tagTemplate = getTagTemplate(bind, state);
            if (!tagTemplate) {
                bind.emitOpError("invalid captured invariant tag binding");
                hadError = true;
                continue;
            }
            state.boundMemRefs[bind.getTarget()] = std::move(*tagTemplate);
            eraseOps.push_back(bind);
            continue;
        }

        // ROCDL raw loads retain the same operand order as the AveLang op:
        // descriptor, vindex, soffset, auxiliary flags.  Decode them before
        // generic memory handling so descriptor provenance is not lost.
        if (isRawBufferLoadOperation(op)) {
            if (op->getNumOperands() < 3 || op->getNumResults() != 1) {
                op->emitOpError("unsupported raw-buffer load signature in invariant validation");
                hadError = true;
                continue;
            }
            auto key = rawBufferMemoryKey(op->getOperand(0), op->getOperand(1),
                                          op->getOperand(2), state);
            auto bytes = buildTagForRawBufferLoad(
                op->getOperand(0), op->getOperand(1), op->getOperand(2),
                op->getResult(0).getType(), state);
            if (key) {
                for (size_t byte = 0; byte < bytes.size(); ++byte) {
                    auto stored = state.memoryTags.find(*key + ":" + std::to_string(byte));
                    if (stored != state.memoryTags.end()) bytes[byte] = stored->second;
                }
            }
            state.valueTags[op->getResult(0)] = std::move(bytes);
            continue;
        }

        if (isRawBufferStoreOperation(op)) {
            // Raw stores are (data, descriptor, vindex, soffset, aux).
            if (op->getNumOperands() < 4) {
                op->emitOpError("unsupported raw-buffer store signature in invariant validation");
                hadError = true;
                continue;
            }
            auto key = rawBufferMemoryKey(op->getOperand(1), op->getOperand(2),
                                          op->getOperand(3), state);
            auto bytes = valueBytes(op->getOperand(0), state);
            if (key) {
                for (size_t byte = 0; byte < bytes.size(); ++byte)
                    state.memoryTags[*key + ":" + std::to_string(byte)] = bytes[byte];
            }
            continue;
        }

        // raw_buffer_load_x1_lds is materialised as a helper call late in the
        // pipeline.  Its first operand is the raw descriptor and its second
        // is the LDS byte pointer, so this is a raw read followed by a typed
        // shared-memory write in the symbolic tag state.
        if (isRawBufferLoadLdsCall(op)) {
            if (op->getNumOperands() != 4) {
                op->emitOpError("unsupported raw-buffer-to-LDS signature in invariant validation");
                hadError = true;
                continue;
            }
            auto key = pointerMemoryKey(op->getOperand(1), state);
            auto bytes = buildTagForRawBufferLoad(op->getOperand(0),
                                                  op->getOperand(2),
                                                  op->getOperand(3),
                                                  mlir::IntegerType::get(op->getContext(), 32),
                                                  state);
            if (key) {
                for (size_t byte = 0; byte < bytes.size(); ++byte)
                    state.memoryTags[*key + ":" + std::to_string(byte)] = bytes[byte];
            }
            continue;
        }

        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
            if (auto tag = buildTagForLoad(load.getMemref(), load.getIndices(),
                                           load.getResult().getType(), state)) {
                state.valueTags[load.getResult()] = *tag;
            } else {
                state.valueTags.erase(load.getResult());
            }
            continue;
        }

        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
            auto key = memoryKey(store.getMemref(), store.getIndices(), state);
            auto width = getByteWidth(store.getValue().getType());
            auto boundIt = state.boundMemRefs.find(store.getMemref());
            if (boundIt != state.boundMemRefs.end()) {
                auto tag = instantiateTag(boundIt->second, store.getIndices(), state);
                if (width && tag.known) {
                    auto bytes = expandTagToBytes(tag, store.getValue().getType());
                    for (size_t byte = 0; byte < bytes.size(); ++byte)
                        state.memoryTags[key + ":" + std::to_string(byte)] =
                            bytes[byte];
                    continue;
                }
            }

            state.boundMemRefs.erase(store.getMemref());
            auto bytes = valueBytes(store.getValue(), state);
            if (width && bytes.size() == *width) {
                for (size_t byte = 0; byte < *width; ++byte)
                    state.memoryTags[key + ":" + std::to_string(byte)] =
                        bytes[byte];
            } else if (width) {
                for (size_t byte = 0; byte < *width; ++byte)
                    state.memoryTags.erase(key + ":" + std::to_string(byte));
            }
            continue;
        }

        // This is the control-flow join used by the canonical OOB-zeroing
        // pattern.  It is deliberately path-insensitive: both alternatives
        // contribute to the per-byte lattice join.
        if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
            auto trueBytes = valueBytes(select.getTrueValue(), state);
            auto falseBytes = valueBytes(select.getFalseValue(), state);
            if (trueBytes.size() != falseBytes.size()) {
                state.valueTags[select.getResult()] =
                    llvm::SmallVector<ByteTag>(bottomBytes(select.getResult().getType()).size(),
                                               ByteTag{ByteTagKind::Top, {}});
            } else {
                llvm::SmallVector<ByteTag> merged;
                merged.reserve(trueBytes.size());
                for (auto [lhs, rhs] : llvm::zip(trueBytes, falseBytes))
                    merged.push_back(mergeByteTag(lhs, rhs));
                state.valueTags[select.getResult()] = std::move(merged);
            }
            continue;
        }

        // llvm.amdgcn.perm selects each output byte from the concatenation of
        // its two i32 inputs. Model the hardware selector rather than treating
        // the operation as opaque arithmetic.
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
            call && call.getCallee() ==
                        "_avelang_amdgpu_llvm_amdgcn_perm" &&
            call.getNumOperands() == 3 && call.getNumResults() == 1) {
            auto control = getConstantInt(call.getOperand(2));
            auto lhsBytes = valueBytes(call.getOperand(0), state);
            auto rhsBytes = valueBytes(call.getOperand(1), state);
            llvm::SmallVector<ByteTag> result;
            result.assign(4, ByteTag{ByteTagKind::Top, {}});
            if (control && lhsBytes.size() == 4 && rhsBytes.size() == 4) {
                const uint32_t selector = static_cast<uint32_t>(*control);
                for (unsigned byte = 0; byte < 4; ++byte) {
                    const unsigned source = (selector >> (byte * 8)) & 0xffu;
                    if (source < 4)
                        result[byte] = lhsBytes[source];
                    else if (source < 8)
                        result[byte] = rhsBytes[source - 4];
                }
            }
            state.valueTags[call.getResult(0)] = std::move(result);
            continue;
        }

        // A scalar bitcast preserves the ordered byte provenance exactly.
        if (auto bitcast = mlir::dyn_cast<mlir::arith::BitcastOp>(op)) {
            auto valueIt = state.valueTags.find(bitcast.getIn());
            auto sourceWidth = getByteWidth(bitcast.getIn().getType());
            auto targetWidth = getByteWidth(bitcast.getOut().getType());
            if (valueIt != state.valueTags.end() && sourceWidth && targetWidth &&
                *sourceWidth == *targetWidth) {
                state.valueTags[bitcast.getOut()] = valueIt->second;
            } else {
                state.valueTags.erase(bitcast.getOut());
            }
            continue;
        }

        if (auto reset = mlir::dyn_cast<TagResetOp>(op)) {
            std::string baseKey = llvm::formatv(
                "p{0}",
                reinterpret_cast<uintptr_t>(reset.getTarget().getAsOpaquePointer()))
                                      .str();
            llvm::SmallVector<std::string> toErase;
            for (const auto &it : state.memoryTags) {
                llvm::StringRef key = it.first();
                if (key == baseKey || key.starts_with(baseKey + "|")) {
                    toErase.push_back(key.str());
                }
            }
            for (const auto &key : toErase) {
                state.memoryTags.erase(key);
            }
            state.boundMemRefs.erase(reset.getTarget());
            eraseOps.push_back(reset);
            continue;
        }

        if (auto assertEq = mlir::dyn_cast<TagAssertEqOp>(op)) {
            auto lhsBytes = valueBytes(assertEq.getLhs(), state);
            auto rhsBytes = valueBytes(assertEq.getRhs(), state);

            if (lhsBytes.size() != rhsBytes.size()) {
                assertEq.emitOpError("tag assertions require equal byte widths");
                hadError = true;
                continue;
            }

            bool proven = true;
            llvm::SmallVector<std::pair<TagValue, TagValue>> unresolvedBytes;
            llvm::SmallVector<std::string> validityConditions;
            for (auto [lhsByte, rhsByte] : llvm::zip(lhsBytes, rhsBytes)) {
                if (lhsByte.kind != ByteTagKind::Concrete ||
                    rhsByte.kind != ByteTagKind::Concrete) {
                    proven = lhsByte.kind == ByteTagKind::Bottom &&
                             rhsByte.kind == ByteTagKind::Bottom;
                } else if (!tagsStructurallyEqual(lhsByte.concrete,
                                                   rhsByte.concrete)) {
                    unresolvedBytes.emplace_back(lhsByte.concrete,
                                                 rhsByte.concrete);
                }
                if (!lhsByte.validityCondition.empty())
                    validityConditions.push_back(lhsByte.validityCondition);
                if (!rhsByte.validityCondition.empty() &&
                    rhsByte.validityCondition != lhsByte.validityCondition)
                    validityConditions.push_back(rhsByte.validityCondition);
                if (!proven) break;
            }

            if (!proven) {
                assertEq.emitOpError("unable to prove invariant tag equality");
                hadError = true;
                continue;
            }

            if (unresolvedBytes.empty() && validityConditions.empty()) {
                eraseOps.push_back(assertEq);
            } else {
                pendingAssertions.push_back(
                    {assertEq, std::move(unresolvedBytes),
                     std::move(validityConditions)});
            }
        }
    }

    if (hadError) {
        return mlir::failure();
    }

    // All active assertions share one SMT model and one solver invocation.
    // The negation of their conjunction is satisfiable exactly when at least
    // one asserted byte equality has a counterexample.
    llvm::SmallVector<std::pair<TagValue, TagValue>> allBytePairs;
    llvm::SmallVector<std::string> allValidityConditions;
    for (const auto &assertion : pendingAssertions)
        allBytePairs.append(assertion.bytePairs.begin(), assertion.bytePairs.end());
    for (const auto &assertion : pendingAssertions)
        allValidityConditions.append(assertion.validityConditions.begin(),
                                     assertion.validityConditions.end());
    if (!allBytePairs.empty() || !allValidityConditions.empty()) {
        auto smt = buildEqualityQuery(allBytePairs, allValidityConditions, state);
        auto z3Response = smt ? runZ3SMTLIB(*smt)
                              : llvm::Expected<Z3Response>(llvm::createStringError(
                                    llvm::inconvertibleErrorCode(),
                                    "could not encode invariant equality"));
        const bool proven = z3Response && z3Response->result == Z3Result::Unsat;
        if (!z3Response) llvm::consumeError(z3Response.takeError());
        if (!proven) {
            for (auto &assertion : pendingAssertions)
                assertion.op.emitOpError("unable to prove invariant tag equality");
            return mlir::failure();
        }
        for (const auto &assertion : pendingAssertions)
            eraseOps.push_back(assertion.op);
    }

    for (auto *op : eraseOps) {
        op->erase();
    }
    return mlir::success();
}

class LateValidateInvariantTagsPass
    : public mlir::PassWrapper<LateValidateInvariantTagsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LateValidateInvariantTagsPass)

    explicit LateValidateInvariantTagsPass(bool validateInvariants)
        : validateInvariants_(validateInvariants) {}

    llvm::StringRef getArgument() const final {
        return "late-validate-invariant-tags";
    }

    llvm::StringRef getDescription() const final {
        return "Late validation of invariant tags using symbolic equality";
    }

    void runOnOperation() final {
        if (!validateInvariants_) {
            getOperation()->walk([](TagAssertEqOp op) { op.erase(); });
            getOperation()->walk([](TagResetOp op) { op.erase(); });
            getOperation()->walk([](TagBindOp op) { op.erase(); });
            return;
        }
        bool failed = false;
        getOperation()->walk([&](mlir::Operation *op) {
            if (auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
                if (mlir::failed(validateFunction(func))) {
                    failed = true;
                    return mlir::WalkResult::interrupt();
                }
            }
            return mlir::WalkResult::advance();
        });

        if (failed) {
            signalPassFailure();
        }
    }

  private:
    bool validateInvariants_;
};

} // namespace

std::unique_ptr<mlir::Pass>
createLateValidateInvariantTagsPass(bool validateInvariants) {
    return std::make_unique<LateValidateInvariantTagsPass>(validateInvariants);
}

} // namespace causalflow::avelang::analysis
