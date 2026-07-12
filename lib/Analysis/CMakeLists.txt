add_library(ave-lang-analysis
  invariant_slice.cc
  invariant_z3.cc
  late_validate_invariant_tags_pass.cc
  validate_invariant_tags_pass.cc
)

target_include_directories(ave-lang-analysis PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/lib/Dialect/AveLang/IR
  ${CMAKE_BINARY_DIR}/lib/Dialect/AveLang/IR
)

target_link_libraries(ave-lang-analysis
  PUBLIC
    ave-lang-dialect
    ave-lang-mlir
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRVectorDialect
    MLIRMemRefDialect
)
