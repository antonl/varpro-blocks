# this is for future standalone executable, if I end up needing it

#cython_add_standalone_executable( varpro MAIN_MODULE varpro.py
#    ${VARPRO_BLOCKS_SOURCE_DIR}/src/blocks.pyx
#    varpro.py
#  )

# Add c module
add_executable( test_arma main.cc )
add_dependencies( test_arma varpro-blocks )
target_link_libraries( test_arma varpro-blocks ${ARMADILLO_LIBRARIES})
set_property(TARGET test_arma PROPERTY CXX_STANDARD 11)

add_executable( test_templating test_templating.cpp )
set_property(TARGET test_templating PROPERTY CXX_STANDARD 11)
