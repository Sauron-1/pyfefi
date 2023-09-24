set(MOD_PREFIX pyfefi.coords.lib)

add_submodule(${MOD_PREFIX}.cartesian_mod cartesian_mod_core.cpp)
add_submodule(${MOD_PREFIX}.sphere_mod sphere_mod_core.cpp)
