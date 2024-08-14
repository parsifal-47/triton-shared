#ifndef TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_H
#define TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonToLinearAlgebraSubprograms/Passes.h.inc"

void populateTritonToLinearAlgebraSubprogramsConversionPatterns(bool pidsToFuncArgs,
                                                   bool addptrToLinalg,
                                                   bool assertToCf,
                                                   RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinearAlgebraSubprogramsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_H
