include_guard(GLOBAL)

set(target_name distributed_mcts)

set(PROJECT_FILES
${CMAKE_CURRENT_SOURCE_DIR}/log.hpp
${CMAKE_CURRENT_SOURCE_DIR}/mcts.hpp
${CMAKE_CURRENT_SOURCE_DIR}/mcts.cpp
${CMAKE_CURRENT_SOURCE_DIR}/neural_network.hpp
${CMAKE_CURRENT_SOURCE_DIR}/neural_network.cpp
${CMAKE_CURRENT_SOURCE_DIR}/threadpool.hpp
${CMAKE_CURRENT_SOURCE_DIR}/waiting_queue.hpp
${CMAKE_CURRENT_SOURCE_DIR}/locked_ptr.hpp
${CMAKE_CURRENT_SOURCE_DIR}/kinarow.hpp
${CMAKE_CURRENT_SOURCE_DIR}/kinarow.cpp
)



set(swig_file ${CMAKE_CURRENT_SOURCE_DIR}/mcts.i)

# when issues with torch libraries paths change enviroment variable xD

message("${TORCH_LIBRARIES}")

#add_library(${target_name} SHARED ${PROJECT_FILES})
set_property(SOURCE ${swig_file} PROPERTY CPLUSPLUS ON)

set(deps extern_spdlog)

swig_add_library(${target_name} TYPE SHARED LANGUAGE python SOURCES 
	${swig_file} ${PROJECT_FILES})
swig_link_libraries(${target_name} ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} ${deps})

set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX  "_d")

add_library(mcts_internal STATIC ${PROJECT_FILES})
target_link_libraries(mcts_internal ${TORCH_LIBRARIES} ${deps})

target_compile_definitions(mcts_internal  PUBLIC ZERO_USE_CUDA)
target_compile_definitions(${target_name} PUBLIC ZERO_USE_CUDA)

target_include_directories(${target_name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(mcts_internal  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

#add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target_name}> ${SWIG_OUTPUT_DIR} )
add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.py" ${SWIG_OUTPUT_DIR} )
add_custom_command(TARGET ${target_name} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/" ${SWIG_OUTPUT_DIR})
# only copy release files because most of the time debug symbols aren't there

include(test)
add_zero_test(mcts_internal "mcts_internal")
add_zero_test(kinarow_get_winner "mcts_internal")
add_zero_test(mcts_race_conditions "mcts_internal")
add_zero_test(mcts_stress_conditions "mcts_internal")
