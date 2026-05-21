set(TARGET llama-speculative-simple)
add_executable(${TARGET}
    speculative-simple.cpp
    ${CMAKE_SOURCE_DIR}/dflash-gpu-log.cpp
)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
