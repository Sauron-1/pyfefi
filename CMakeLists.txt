cmake_minimum_required(VERSION 3.11)
project(pyfefi)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MARCH "native" CACHE STRING "Architecture to tell gcc to optimize for (-march)")
option(USE_OPENMP ON)

message("-- Compile for CPU ${MARCH}")

find_package(pybind11 QUIET)
if (NOT pybind11_FOUND)
    message("-- pybind11 not found. Downloading...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL    https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
        URL_HASH MD5=671deeeaccfccd7c0389e43237c71cf3
        TLS_VERIFY true
        )
    FetchContent_MakeAvailable(pybind11)
else()
    message("-- Found pybind11 ${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
endif()

find_package(tuple_arithmetic QUIET)
if (NOT tuple_arithmetic_FOUND)
    message("-- tuple_arithmetic not found. Downloading...")
    include(FetchContent)
    FetchContent_Declare(
        tuple_arithmetic
        GIT_REPOSITORY https://github.com/Sauron-1/tuple_arithmetic.git
        GIT_TAG        547ebd0b978d6542431363c4f5460b58cc708ef6
        )
    set(BULID_TESTS_TMP ${BUILD_TESTS})
    set(BUILD_TESTS OFF)
    FetchContent_MakeAvailable(tuple_arithmetic)
    set(BUILD_TESTS ${BULID_TESTS_TMP})
else()
    message("-- Found tuple_arithmetic ${tuple_arithmetic_VERSION}: ${tuple_arithmetic_INCLUDE_DIRS}")
endif()

find_package(picinterp QUIET)
if (NOT picinterp_FOUND)
    message("-- picinterp not found. Downloading...")
    include(FetchContent)
    FetchContent_Declare(
        picinterp
        GIT_REPOSITORY https://github.com/Sauron-1/picinterp.git
        GIT_TAG        63d2955a2305229146997d7b5320fed93cf0d932
        )
    set(BULID_TESTS_TMP ${BUILD_TESTS})
    set(BUILD_TESTS OFF)
    FetchContent_MakeAvailable(picinterp)
    set(BUILD_TESTS ${BULID_TESTS_TMP})
else()
    message("-- Found picinterp ${picinterp_VERSION}: ${picinterp_INCLUDE_DIRS}")
endif()

if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    message("-- OpenMP found, version ${OpenMP_CXX_VERSION}")
else()
    message("-- OpenMP is disabled.")
endif()

function(add_submodule module)
    set(target_name ${module})
    string(REPLACE "." "/" modpath ${module})
    string(REPLACE "." ";" modlist ${module})

    list(GET modlist -1 modname)
    list(REMOVE_AT modlist -1)
    list(REMOVE_AT ARGV 0)

    if(modlist)
        string(REPLACE ";" "/" relpath "${modlist}")
    else()
        set(relpath "")
    endif()
    pybind11_add_module(${target_name} ${ARGV})
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${target_name} PRIVATE tuple_arithmetic)
    target_link_libraries(${target_name} PRIVATE picinterp)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target_name} PRIVATE /O2 /fp:fast /DNDEBUG)
    else()
        target_compile_options(${target_name} PRIVATE -O3 -ffast-math -DNDEBUG -march=${MARCH})
        #target_compile_options(${target_name} PRIVATE -O3 -ffast-math -march=${MARCH} -DBOUNDSCHECK -g -Wall -Wextra -Wno-sign-compare)
    endif()

    if (USE_OPENMP)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
        set(outdir ${relpath})
    else()
        set(outdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${relpath})
    endif()
    set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${outdir})
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${modname})
endfunction()

add_subdirectory(src)
