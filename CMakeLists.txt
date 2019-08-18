# Copyright Tom Westerhout (c) 2019
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0074 NEW)

# detect if we're being bundled, disable test and install targets in that case
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
endif()

project(dual-annealing)

include(CTest)
include(CheckCXXCompilerFlag)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to RelWithDebInfo, use \
-DCMAKE_BUILD_TYPE= to override."
    )
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _DA_BUILD_TYPE)

# By default, we only enable asserts in Debug builds
if("${_DA_BUILD_TYPE}" STREQUAL "DEBUG")
    set(_DA_ASSERTS_ARE_OKAY ON)
else()
    set(_DA_ASSERTS_ARE_OKAY OFF)
endif()

if(BUILD_TESTING AND NOT_SUBPROJECT)
    set(_DA_TESTS_ARE_OKAY ON)
    set(_DA_EXAMPLES_ARE_OKAY ON)
else()
    set(_DA_TESTS_ARE_OKAY OFF)
    set(_DA_EXAMPLES_ARE_OKAY ON)
endif()

option(DA_USE_VALGRIND "Perform tests with Valgrind" OFF)
option(DA_USE_BLAS "Use BLAS for 1D vector operations" OFF)
option(DA_BUILD_TESTING "Build tests" ${_DA_TESTS_ARE_OKAY})
option(DA_BUILD_EXAMPLES "Build examples" ${_DA_EXAMPLES_ARE_OKAY})
option(DA_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(DA_ENABLE_COVERAGE "Generate coverage for codecov.io" OFF)
option(DA_INSTALL_DOCS "Install documentation alongside library" ON)
option(DA_DEBUG "Include assertions" ${_DA_ASSERTS_ARE_OKAY})


# Warnings
################################################################################
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(DA_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-c++14-compat -Wno-unused -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-padded
        )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(DA_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused)
endif()

# Dependencies
################################################################################
find_package(Git)
if(GIT_FOUND)
    # Update submodules
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/pcg-cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/gsl-lite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # Temporary only!
    # execute_process(
    #     COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/lbfgs-cpp
    #     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if (DA_BUILD_TESTING)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/Catch2
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
else()
    message(WARNING "Git not found so submodules were not updated. Expect problems")
endif()

add_library(pcg INTERFACE)
target_include_directories(pcg INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pcg-cpp/include)

if (NOT TARGET gsl::gsl-lite)
    add_subdirectory(third_party/gsl-lite)
endif()

set(LBFGS_DEBUG OFF)
add_subdirectory(third_party/lbfgs-cpp)

if (DA_BUILD_TESTING)
    if (NOT TARGET Catch2::Catch2)
        option(CATCH_BUILD_TESTING OFF)
        option(CATCH_BUILD_EXAMPLES OFF)
        option(CATCH_BUILD_EXTRA_TESTS OFF)
        add_subdirectory(third_party/Catch2)
    endif()
    enable_testing()
endif()


add_library(Common INTERFACE)
target_include_directories(Common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(Common INTERFACE cxx_std_17)
target_compile_options(Common INTERFACE "${DA_WARNING_FLAGS}")
target_link_libraries(Common INTERFACE pcg)
if (DA_DEBUG)
    target_compile_definitions(Common INTERFACE DA_DEBUG=1)
endif()
if (DA_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(Common INTERFACE DA_USE_BLAS=1)
    target_link_libraries(Common INTERFACE ${BLAS_LIBRARIES})
endif()

# CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_NATIVE_SUPPORTED)
# if(LBFGS_COMPILER_OPT_NATIVE_SUPPORTED)
#     message(STATUS "[lbfgs-cpp] Using -march=native.")
#     target_compile_options(Common INTERFACE "-march=native")
# endif()

# CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" LBFGS_COMPILER_OPT_HIDDEN_SUPPORTED)
# if(LBFGS_COMPILER_OPT_HIDDEN_SUPPORTED)
#     message(STATUS "[lbfgs-cpp] Using -fvisibility=hidden.")
#     target_compile_options(Common INTERFACE "-fvisibility=hidden")
# endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(Common INTERFACE -fcolor-diagnostics)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(Common INTERFACE -fdiagnostics-color=always)
endif()


add_library(dual_annealing SHARED
    include/config.hpp
    include/assert.hpp
    include/buffers.hpp
    include/tsallis_distribution.hpp
    include/chain.hpp
    src/assert.cpp
    src/buffers.cpp)
target_link_libraries(dual_annealing PUBLIC Common gsl::gsl-lite lbfgs-cpp::lbfgs)

# if (TRUE)
#     target_compile_options(line_search PUBLIC "-fprofile-instr-generate" "-fcoverage-mapping")
#     target_link_libraries(line_search PUBLIC "-fprofile-instr-generate" "-fcoverage-mapping")
# endif()

if (DA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (DA_BUILD_TESTING)
    add_subdirectory(test)
endif()

if (DA_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

