include_guard(GLOBAL)

set(target_name distributed_mcts)

set(PROJECT_FILES
${CMAKE_CURRENT_SOURCE_DIR}/mcts.hpp
${CMAKE_CURRENT_SOURCE_DIR}/mcts.cpp
${CMAKE_CURRENT_SOURCE_DIR}/neural_network.hpp
${CMAKE_CURRENT_SOURCE_DIR}/neural_network.cpp
${CMAKE_CURRENT_SOURCE_DIR}/threadpool.hpp
${CMAKE_CURRENT_SOURCE_DIR}/kinarow.hpp
${CMAKE_CURRENT_SOURCE_DIR}/kinarow.cpp
)

set(swig_file ${CMAKE_CURRENT_SOURCE_DIR}/mcts.i)

file(TO_NATIVE_PATH ${TORCH_LIBRARIES} ${TORCH_LIBRARIES})


#add_library(${target_name} SHARED ${PROJECT_FILES})
set_property(SOURCE ${swig_file} PROPERTY CPLUSPLUS ON)
swig_add_library(${target_name} TYPE SHARED LANGUAGE python SOURCES 
	${swig_file} ${PROJECT_FILES} ${TORCH_LIBRARIES})

target_include_directories(${target_name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)