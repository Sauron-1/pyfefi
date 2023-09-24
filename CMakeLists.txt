cmake_minimum_required(VERSION 3.11)
project(pyfefi)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MARCH "native" CACHE STRING "Architecture to tell gcc to optimize for (-march)")

message("-- Compile for CPU ${MARCH}")

find_package(pybind11 QUIET)
if (NOT pybind11_FOUND)
    message("-- pybind11 not found. Downloading...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL    https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip
        URL_HASH MD5=c62d9e05243bd31cdb3bae1bb2f56655
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
        GIT_TAG        master
        )
    set(BULID_TESTS_TMP ${BUILD_TESTS})
    set(BUILD_TESTS OFF)
    FetchContent_MakeAvailable(tuple_arithmetic)
    set(BUILD_TESTS ${BULID_TESTS_TMP})
else()
    message("-- Found tuple_arithmetic ${tuple_arithmetic_VERSION}: ${tuple_arithmetic_INCLUDE_DIRS}")
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
    target_compile_options(${target_name} PRIVATE -O3 -ffast-math -DNDEBUG -march=${MARCH})

    if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
        set(outdir ${relpath})
    else()
        set(outdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${relpath})
    endif()
    set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${outdir})
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${modname})
endfunction()

add_subdirectory(src)
