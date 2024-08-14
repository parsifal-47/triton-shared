#ifndef TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES
#define TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinearAlgebraSubprograms : Pass<"triton-to-linear-algebra-subprograms", "mlir::ModuleOp"> {
  let summary = "Convert Linalg operations to library calls";
}

#endif
