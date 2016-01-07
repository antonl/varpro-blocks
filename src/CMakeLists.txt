# If the pyx file is a C++ file, we should specify that here.
#set_source_files_properties(
#    ${VARPRO_BLOCKS_SOURCE_DIR}/src/varprocpp.pyx
#    ${VARPRO_BLOCKS_SOURCE_DIR}/src/arma.pyx
#    PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
#cython_add_module( varproold varproold.pyx )
#cython_add_module( arma arma.pyx )
#cython_add_module( varprocpp varprocpp.pyx )

#add_definitions(-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION) # get rid of annoying api message

# as discussed here,
# http://stackoverflow.com/questions/12253389/numpy-c-api-link-several-object-files
# http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api:
#add_definitions(-D PY_ARRAY_UNIQUE_SYMBOL=arma_NUMPY_API)

add_library(varpro SHARED varpro_module.cpp varpro_objects.cpp varpro_util.cpp)
set_target_properties(varpro PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 11)
set_target_properties(varpro PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(varpro ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})
if(APPLE)
    # following advice from http://blog.tim-smith.us/2015/09/python-extension-modules-os-x/
    set_target_properties(varpro PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    set_target_properties(varpro PROPERTIES MACOSX_RPATH TRUE)
endif() 
