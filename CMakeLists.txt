# Defines the CMake commands/policies
cmake_minimum_required( VERSION 2.8.5 )

# Set the project name
project( VARPRO_BLOCKS )

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

# We need the numpy includes, so add them to the source
find_package( NumPy REQUIRED )

find_package( Armadillo REQUIRED )

set( PYCXX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pycxx" CACHE FILEPATH
    "location of PyCXX distribution dir" )
find_package (PyCXX REQUIRED )

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories( ${VARPRO_BLOCKS_SOURCE_DIR}/include ${NUMPY_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${PYCXX_INCLUDE_DIR} )

enable_testing()
find_file( PYTEST_EXECUTABLE py.test )
add_custom_target(check ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
        ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --verbose
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test( pytest "${PYTEST_EXECUTABLE}" -vv )

# Process the CMakeLists.txt in the 'src' and 'bin' directory.
add_subdirectory( src )
add_subdirectory( bin )

