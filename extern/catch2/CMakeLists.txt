include_guard(GLOBAL)
set(target_name "extern_catch2")
add_library(${target_name} INTERFACE)
target_include_directories(${target_name} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")


