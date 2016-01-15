# Defines the CMake commands/policies
cmake_minimum_required( VERSION 3.1 )

project( VARPRO_BLOCKS )

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )
set(SPDLOG_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/extern/spdlog/include)
set(PYBIND11_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include)
set(ARMADILLO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/extern/armadillo/include)
# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
#include( UseCython )

# We need the numpy includes, so add them to the source
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package( PythonLibs REQUIRED )
find_package( NumPy REQUIRED )

set(MKL_ROOT "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64") 
set(BLAS_LIBRARIES "")
set(LAPACK_LIBRARIES ${MKL_ROOT}/mkl_rt.lib)
set(MKL_INCLUDE_DIRS ${MKL_ROOT}/../include)

set(Boost_NO_SYSTEM_PATHS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ALL_DYN_LINK OFF)
find_package( boost REQUIRED COMPONENTS math)


include_directories( 
    ${VARPRO_BLOCKS_SOURCE_DIR}/include 
    ${ARMADILLO_INCLUDE_DIRS} 
    ${PYTHON_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIRS}
    )

enable_testing()
find_file( PYTEST_EXECUTABLE py.test )
add_custom_target(check ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
        ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --verbose
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test( pytest "${PYTEST_EXECUTABLE}" -vv )

# Process the CMakeLists.txt in the 'src' and 'bin' directory.
add_subdirectory( src )
add_subdirectory( bin )

# Create a dist folder. This is where you'd run python setup.py install
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/dist)
install(FILES setup.py README.md DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY examples src DESTINATION ${CMAKE_INSTALL_PREFIX}
    FILES_MATCHING PATTERN "*.py" PATTERN "*.ipynb")
# install targets generated from src in the CMakeLists.txt in that directory
