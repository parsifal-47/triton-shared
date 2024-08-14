#ifndef TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES_H
#define TRITON_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonToLinearAlgebraSubprograms/TritonToLinearAlgebraSubprograms.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToLinearAlgebraSubprograms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
