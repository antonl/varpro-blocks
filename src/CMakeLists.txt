# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
    ${VARPRO_BLOCKS_SOURCE_DIR}/src/varprocpp.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( varpro varpro.pyx )
cython_add_module( varprocpp varprocpp.pyx )

add_library(varpro-blocks STATIC varpro-block.cpp)
set_property(TARGET varpro-blocks PROPERTY CXX_STANDARD 11)
set_property(TARGET varprocpp PROPERTY CXX_STANDARD 11)
