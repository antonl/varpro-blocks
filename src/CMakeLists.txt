# If the pyx file is a C++ file, we should specify that here.
#set_source_files_properties(
#    ${VARPRO_BLOCKS_SOURCE_DIR}/src/varprocpp.pyx
#    ${VARPRO_BLOCKS_SOURCE_DIR}/src/arma.pyx
#    PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
#cython_add_module( varpro varpro.pyx )
#cython_add_module( arma arma.pyx )
#cython_add_module( varprocpp varprocpp.pyx )

add_definitions(-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION) # get rid of annoying api message
add_library(arma SHARED arma_objects.cpp arma_module.cpp ${PYCXX_SOURCES} )
set_target_properties(arma PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(arma PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(arma PROPERTIES MACOSX_RPATH TRUE)
if(APPLE)
    # following advice from http://blog.tim-smith.us/2015/09/python-extension-modules-os-x/
    set_target_properties(arma PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif() 

add_library(varpro-blocks STATIC varpro-block.cpp)
set_property(TARGET varpro-blocks PROPERTY CXX_STANDARD 11)
set_property(TARGET arma PROPERTY CXX_STANDARD 11)
#set_property(TARGET varprocpp PROPERTY CXX_STANDARD 11)
#set_property(TARGET arma PROPERTY CXX_STANDARD 11)
