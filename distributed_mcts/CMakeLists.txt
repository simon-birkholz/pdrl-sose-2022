include_guard(GLOBAL)

set(target_name distributed_mcts)

set(PROJECT_FILES
${CMAKE_CURRENT_SOURCE_DIR}/mcts.hpp
${CMAKE_CURRENT_SOURCE_DIR}/mcts.cpp
${CMAKE_CURRENT_SOURCE_DIR}/neural_network.hpp
${CMAKE_CURRENT_SOURCE_DIR}/neural_network.cpp
${CMAKE_CURRENT_SOURCE_DIR}/threadpool.hpp
)

set(swig_file ${CMAKE_CURRENT_SOURCE_DIR}/mcts.i)

#add_library(${target_name} SHARED ${PROJECT_FILES})

swig_add_library(${target_name} TYPE SHARED LANGUAGE python SOURCES ${swig_file} ${PROJECT_FILES})

target_include_directories(${target_name} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)