# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
    ${VARPRO_BLOCKS_SOURCE_DIR}/src/blocks.pyx
    PROPERTIES CYTHON_IS_CXX FALSE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( _blocks blocks.pyx )
cython_add_module( _varpro varpro.pyx )

