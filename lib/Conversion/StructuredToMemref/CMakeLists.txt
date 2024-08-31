add_triton_library(StructuredToMemref
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp

  DEPENDS
  StructuredToMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTensorTransformOps
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
)
