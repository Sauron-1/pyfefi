cmake_minimum_required(VERSION 3.11)
project(pyfefi)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MARCH "native" CACHE STRING "Architecture to tell gcc to optimize for (-march)")
option(USE_OPENMP ON)

message("-- Compile for CPU ${MARCH}")

add_subdirectory(thirdparty/pybind11)
add_subdirectory(thirdparty/tuple_arithmetic)
add_subdirectory(thirdparty/picinterp)

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
