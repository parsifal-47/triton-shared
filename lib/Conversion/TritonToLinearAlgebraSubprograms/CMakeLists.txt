add_triton_library(TritonToLinearAlgebraSubprograms
  TritonToLinearAlgebraSubprogramsPass.cpp

  DEPENDS
  TritonToLinearAlgebraSubprogramsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
)
