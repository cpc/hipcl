# Test symbol access

add_hipcl_binary(
	hipModuleLoad
	main.cpp)

add_hipcl_device_binary(
	hipModuleLoadBinary
	kernel.cpp)

add_dependencies(hipModuleLoad hipModuleLoadBinary)

add_test(NAME hipModuleLoad
	 COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipModuleLoad"
	 )

set_tests_properties(hipModuleLoad PROPERTIES
	PASS_REGULAR_EXPRESSION PASSED)

