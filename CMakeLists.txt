cmake_minimum_required(VERSION 3.18)
project(community_detection_cpp VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(COMMUNITY_DETECTION_BUILD_TESTS "Build tests" ON)
option(COMMUNITY_DETECTION_BUILD_EXAMPLES "Build examples" ON)
option(COMMUNITY_DETECTION_BUILD_CLI "Build experiment CLI" ON)
option(COMMUNITY_DETECTION_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

add_library(community_detection
    src/core/graph.cpp
    src/core/graph_transform.cpp
    src/core/partition.cpp
    src/io/graph_io.cpp
    src/metrics/modularity.cpp
    src/metrics/partition_metrics.cpp
    src/metrics/external.cpp
    src/algorithms/girvan_newman.cpp
    src/algorithms/spectral.cpp
    src/algorithms/louvain.cpp
    src/algorithms/label_propagation.cpp
    src/experiment/runner.cpp
)
add_library(community_detection::community_detection ALIAS community_detection)

target_include_directories(community_detection
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

function(cd_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    if(COMMUNITY_DETECTION_ENABLE_SANITIZERS AND NOT MSVC)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

cd_set_warnings(community_detection)

install(
    TARGETS community_detection
    EXPORT community_detectionTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/community_detection DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT community_detectionTargets
    FILE community_detectionTargets.cmake
    NAMESPACE community_detection::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/community_detection
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/community_detectionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/community_detectionConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/community_detection
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/community_detectionConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/community_detectionConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/community_detectionConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/community_detection
)
install(
    FILES LICENSE CITATION.cff
    DESTINATION ${CMAKE_INSTALL_DATADIR}/community_detection
)

if(COMMUNITY_DETECTION_BUILD_CLI)
    add_executable(community_detect apps/community_detect.cpp)
    target_link_libraries(community_detect PRIVATE community_detection::community_detection)
    cd_set_warnings(community_detect)
    install(TARGETS community_detect RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(COMMUNITY_DETECTION_BUILD_EXAMPLES)
    add_executable(run_toy_example examples/run_toy_example.cpp)
    target_link_libraries(run_toy_example PRIVATE community_detection::community_detection)
    cd_set_warnings(run_toy_example)
endif()

if(COMMUNITY_DETECTION_BUILD_TESTS)
    include(CTest)
    add_executable(community_detection_tests tests/test_algorithms.cpp)
    target_link_libraries(community_detection_tests PRIVATE community_detection::community_detection)
    cd_set_warnings(community_detection_tests)
    add_test(NAME community_detection_tests COMMAND community_detection_tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND COMMUNITY_DETECTION_BUILD_CLI)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import numpy, scipy, pandas, sklearn"
            RESULT_VARIABLE COMMUNITY_DETECTION_PYTHON_DEPS
            OUTPUT_QUIET
            ERROR_QUIET
        )
    endif()
    if(Python3_Interpreter_FOUND
       AND COMMUNITY_DETECTION_BUILD_CLI
       AND COMMUNITY_DETECTION_PYTHON_DEPS EQUAL 0)
        add_test(
            NAME experiment_pipeline_smoke
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pipeline.py
                --binary $<TARGET_FILE:community_detect>
        )
    endif()
endif()
