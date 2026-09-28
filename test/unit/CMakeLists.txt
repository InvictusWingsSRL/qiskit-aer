cmake_minimum_required(VERSION 3.15)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(aer_native_tests LANGUAGES CXX)
  enable_testing()
  option(AER_TEST_OPENMP "Exercise the OpenMP implementation" ON)
  if(AER_TEST_OPENMP)
    find_package(OpenMP REQUIRED)
  endif()
endif()

add_executable(extended_stabilizer_threading extended_stabilizer_threading.cpp)
target_compile_features(extended_stabilizer_threading PRIVATE cxx_std_17)
target_include_directories(extended_stabilizer_threading PRIVATE ../../src)
if(TARGET AER_DEPENDENCY_PKG::nlohmann_json)
  target_link_libraries(extended_stabilizer_threading PRIVATE
    AER_DEPENDENCY_PKG::nlohmann_json)
else()
  find_path(AER_JSON_INCLUDE_DIR nlohmann/json.hpp)
  if(NOT AER_JSON_INCLUDE_DIR)
    message(FATAL_ERROR "Set AER_JSON_INCLUDE_DIR to the nlohmann/json include directory")
  endif()
  target_include_directories(extended_stabilizer_threading PRIVATE
    "${AER_JSON_INCLUDE_DIR}")
endif()
if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(extended_stabilizer_threading PRIVATE OpenMP::OpenMP_CXX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Discard unrelated, out-of-line Clifford methods from transitive headers.
  # Some of those methods require OpenMP even when _OPENMP is not defined.
  target_compile_options(extended_stabilizer_threading PRIVATE
    -ffunction-sections -fdata-sections)
  if(APPLE)
    target_link_options(extended_stabilizer_threading PRIVATE -Wl,-dead_strip)
  else()
    target_link_options(extended_stabilizer_threading PRIVATE -Wl,--gc-sections)
  endif()
endif()

add_test(NAME extended_stabilizer_threading COMMAND extended_stabilizer_threading)
add_test(NAME extended_stabilizer_thread_limit COMMAND
  "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=256 OMP_THREAD_LIMIT=2
  "$<TARGET_FILE:extended_stabilizer_threading>")
add_test(NAME extended_stabilizer_dynamic_threads COMMAND
  "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=4 OMP_DYNAMIC=TRUE
  "$<TARGET_FILE:extended_stabilizer_threading>")
set_tests_properties(extended_stabilizer_threading extended_stabilizer_thread_limit
  extended_stabilizer_dynamic_threads PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
