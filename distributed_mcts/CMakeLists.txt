include_guard(GLOBAL)

add_library(mcts_core STATIC
${CMAKE_CURRENT_SOURCE_DIR}/evaluator.hpp
${CMAKE_CURRENT_SOURCE_DIR}/kinarow.hpp
${CMAKE_CURRENT_SOURCE_DIR}/kinarow.cpp
${CMAKE_CURRENT_SOURCE_DIR}/mcts.hpp
${CMAKE_CURRENT_SOURCE_DIR}/mcts.cpp
${CMAKE_CURRENT_SOURCE_DIR}/threadpool.hpp
)
target_include_directories(mcts_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# linked into the python module
set_target_properties(mcts_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(mcts_core PUBLIC Threads::Threads)

add_zero_test(kinarow_get_winner mcts_core)
add_zero_test(mcts_search mcts_core)

if (ZERO_BUILD_BINDINGS)
  set(target_name distributed_mcts)
  set(swig_file ${CMAKE_CURRENT_SOURCE_DIR}/mcts.i)
  set_property(SOURCE ${swig_file} PROPERTY CPLUSPLUS ON)

  swig_add_library(${target_name} TYPE SHARED LANGUAGE python SOURCES
    ${swig_file}
    ${CMAKE_CURRENT_SOURCE_DIR}/neural_network.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/neural_network.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/waiting_queue.hpp
  )
  swig_link_libraries(${target_name} mcts_core ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
  set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX "_d")

  add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.py" ${SWIG_OUTPUT_DIR})
  add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target_name}> ${SWIG_OUTPUT_DIR})
endif()
