set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)

execute_process(COMMAND "${LLVM_CONFIG}" "--cmakedir"
		OUTPUT_VARIABLE LLVM_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE RES)

if(NOT RES EQUAL 0)
  message(FATAL_ERROR "failed to run llvm-config (${LLVM_CONFIG})")
endif()

message(STATUS "LLVM CMake directory: ${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG HINTS "${LLVM_DIR}" PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)

######################################

add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options("-fno-rtti")
endif()

if(NOT LLVM_ENABLE_EH)
  add_compile_options("-fno-exceptions")
endif()

add_library(LLVMHipDynMem MODULE HipDynMem.cpp)

install(TARGETS LLVMHipDynMem
        LIBRARY DESTINATION "${HIPCL_LLVM_DIR}"
        ARCHIVE DESTINATION "${HIPCL_LLVM_DIR}"
        )
