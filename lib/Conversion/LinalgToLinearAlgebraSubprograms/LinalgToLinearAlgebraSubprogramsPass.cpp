//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/LinalgToLinearAlgebraSubprograms.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-to-las"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LA_TO_LAS
#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct MatmulConverter : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto doubleType = rewriter.getF64Type();
    auto intType = rewriter.getI32Type();
    auto doublePtrType = PointerType::get(doubleType, ADDRESS_SPACE_GENERIC);

    auto funcType = FunctionType::get(rewriter.getNoneType(),
        {intType, intType, intType, intType, intType, intType, doubleType,
         doublePtrType, intType, doublePtrType, intType, doubleType,
         doublePtrType, intType}, false);

    const char *funcName = "cblas_dgemm";
    auto func = module.lookupSymbol<FuncOp>(funcName);
    if (!func) {
      func = rewriter.create<FuncOp>(loc, funcName, funcType);
    }

    Value A = op.getOperands()[0];
    Value B = op.getOperands()[1];
    Value C = op.getResult(0);

    int64_t M = cast<MemRefType>(A.getType()).getShape()[0];
    int64_t K = cast<MemRefType>(A.getType()).getShape()[1];
    int64_t N = cast<MemRefType>(B.getType()).getShape()[1];

    Value alpha = rewriter.create<arith::ConstantOp>(loc, doubleType, rewriter.getF64FloatAttr(1.0));
    Value beta = rewriter.create<arith::ConstantOp>(loc, doubleType, rewriter.getF64FloatAttr(0.0));

    Value CblasRowMajor = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(101));
    Value CblasNoTrans = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(111));
    Value MVal = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(M));
    Value NVal = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(N));
    Value KVal = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(K));
    Value LDA = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(K));
    Value LDB = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(N));
    Value LDC = rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getI32IntegerAttr(N));

    rewriter.replaceOpWithNewOp<CallOp>(op, func, ValueRange{
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        MVal, NVal, KVal,
        alpha, A, LDA,
        B, LDB, beta,
        C, LDC
    });

    return success();
  }
};

class LinalgToLinearAlgebraSubprogramsPass
    : public triton::impl::LinalgToLinearAlgebraSubprogramsBase<LinalgToLinearAlgebraSubprogramsPass> {
  using LinalgToLinearAlgebraSubprogramsBase<
      LinalgToLinearAlgebraSubprogramsPass>::LinalgToLinearAlgebraSubprogramsBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());

    patterns.add<SplatConverter>(patterns.getContext());

    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createLinalgToLinearAlgebraSubprogramsPass() {
  return std::make_unique<LinalgToLinearAlgebraSubprogramsPass>();
}
