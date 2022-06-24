
function(add_zero_test namer libs)
set(name test_${namer})
	set(cpp test/${namer}.cpp)
	add_executable(${name} ${cpp})
	
	set( _deps extern_catch2 ${libs}) 
	target_link_libraries(${name} ${_deps})
	message("test: ${name} ${namer} ${_deps}")
	if (UNIX)
		#i dont know why this is needed
		target_link_libraries(${name} tbb)
	endif()
	set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
	
	#add_definitions(${name} "-DCATCH_CONFIG_MAIN")
	
	set_target_properties(${name} PROPERTIES
		FOLDER testing/${target_name}
		OUTPUT_NAME testing_${name}
		RUNTIME_OUTPUT_DIRECTORY ${output_dir}
		VS_DEBUGGER_WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}/$<CONFIG>"
		)
	add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${output_dir} )
	
endfunction()
