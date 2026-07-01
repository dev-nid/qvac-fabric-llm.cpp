cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)
cmake_policy(SET CMP0116 NEW)
if (POLICY CMP0147)
    # Parallel build custom build steps
    cmake_policy(SET CMP0147 NEW)
endif()

find_package(Vulkan COMPONENTS glslc REQUIRED)

if (DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()
find_package(SPIRV-Headers CONFIG QUIET)
if (NOT TARGET SPIRV-Headers::SPIRV-Headers)
    # The Android NDK ships no SPIRV-Headers package. Reuse the pinned headers
    # already fetched under the build tree's _deps (offline) for the on-device
    # Vulkan build; fall back to the upstream repo if they are not present.
    include(FetchContent)
    if (EXISTS "${CMAKE_BINARY_DIR}/_deps/spirv_headers-src/CMakeLists.txt")
        FetchContent_Declare(spirv_headers SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/spirv_headers-src")
    else()
        # Pin the exact commit (integrity) rather than the mutable tag; the
        # trailing comment records the corresponding SDK tag.
        FetchContent_Declare(spirv_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
            GIT_TAG        2acb319af38d43be3ea76bfabf3998e5281d8d12) # vulkan-sdk-1.3.290.0
    endif()
    FetchContent_MakeAvailable(spirv_headers)
endif()

# The NDK's find_package(Vulkan) yields C-only headers, so <vulkan/vulkan.hpp>
# must be supplied separately on Android. Reuse the pinned headers already in
# the build tree's _deps; only add the include dir (the Vulkan::Headers target
# already exists from the NDK, so we must not redefine it).
set(GGML_VK_CPP_HEADERS_DIR "")
if (ANDROID)
    if (EXISTS "${CMAKE_BINARY_DIR}/_deps/vulkan_headers-src/include/vulkan/vulkan.hpp")
        set(GGML_VK_CPP_HEADERS_DIR "${CMAKE_BINARY_DIR}/_deps/vulkan_headers-src/include")
    else()
        include(FetchContent)
        # Pinned to the exact commit (integrity) for the SDK tag in the comment.
        # SOURCE_SUBDIR points at a path with no CMakeLists.txt so
        # FetchContent_MakeAvailable only downloads the headers and skips
        # add_subdirectory() — we just need the include dir, and adding
        # Vulkan-Headers' own targets would clash with the NDK's Vulkan::Headers.
        # This also avoids the deprecated FetchContent_Populate(), which errors
        # under CMake 4.x (policy CMP0169).
        FetchContent_Declare(vulkan_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG        b379292b2ab6df5771ba9870d53cf8b2c9295daf # vulkan-sdk-1.3.290.0
            SOURCE_SUBDIR  ggml-vk-headers-only)
        FetchContent_MakeAvailable(vulkan_headers)
        set(GGML_VK_CPP_HEADERS_DIR "${vulkan_headers_SOURCE_DIR}/include")
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Parallel build object files
    add_definitions(/MP)
endif()

# Add options to disable cooperative matrix support
option(GGML_VULKAN_DISABLE_COOPMAT "Disable VK_KHR_cooperative_matrix support" OFF)
option(GGML_VULKAN_DISABLE_COOPMAT2 "Disable VK_NV_cooperative_matrix2 support" OFF)

function(detect_host_compiler)
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        find_program(HOST_C_COMPILER NAMES cl gcc clang NO_CMAKE_FIND_ROOT_PATH)
        find_program(HOST_CXX_COMPILER NAMES cl g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_program(HOST_C_COMPILER NAMES gcc clang NO_CMAKE_FIND_ROOT_PATH)
        find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    endif()
    set(HOST_C_COMPILER "${HOST_C_COMPILER}" PARENT_SCOPE)
    set(HOST_CXX_COMPILER "${HOST_CXX_COMPILER}" PARENT_SCOPE)
endfunction()

# Function to test shader extension support
# Parameters:
#  EXTENSION_NAME - Name of the extension to test (e.g., "GL_EXT_integer_dot_product")
#  TEST_SHADER_FILE - Path to the test shader file
#  RESULT_VARIABLE - Name of the variable to set (ON/OFF) based on test result
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    execute_process(
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_SHADER_FILE}"
        OUTPUT_VARIABLE glslc_output
        ERROR_VARIABLE glslc_error
    )

    if (${glslc_error} MATCHES ".*extension not supported: ${EXTENSION_NAME}.*")
        message(STATUS "${EXTENSION_NAME} not supported by glslc")
        set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    else()
        message(STATUS "${EXTENSION_NAME} supported by glslc")
        set(${RESULT_VARIABLE} ON PARENT_SCOPE)
        add_compile_definitions(${RESULT_VARIABLE})

        # Ensure the extension support is forwarded to vulkan-shaders-gen
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -D${RESULT_VARIABLE}=ON)
        set(VULKAN_SHADER_GEN_CMAKE_ARGS "${VULKAN_SHADER_GEN_CMAKE_ARGS}" PARENT_SCOPE)
    endif()
endfunction()

if (Vulkan_FOUND)
    message(STATUS "Vulkan found")

    ggml_add_backend_library(ggml-vulkan
                             ggml-vulkan.cpp
                             ../../include/ggml-vulkan.h
                            )

    set(VULKAN_SHADER_GEN_CMAKE_ARGS "")

    # Test all shader extensions
    if (NOT GGML_VULKAN_DISABLE_COOPMAT)
        test_shader_extension_support(
            "GL_KHR_cooperative_matrix"
            "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/coopmat.comp"
            "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"
        )
    else()
        message(STATUS "VK_KHR_cooperative_matrix support disabled by GGML_VULKAN_DISABLE_COOPMAT")
    endif()

    if (NOT GGML_VULKAN_DISABLE_COOPMAT2)
        test_shader_extension_support(
            "GL_NV_cooperative_matrix2"
            "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/coopmat2.comp"
            "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
        )
    else()
        message(STATUS "VK_NV_cooperative_matrix2 support disabled by GGML_VULKAN_DISABLE_COOPMAT2")
    endif()

    # qvac: only the FlashAttn MMQ shaders reference FaTypeK / FA_TYPE_* spec
    # constants that weren't ported from upstream b9341 into the qvac fork's
    # macro-based dispatch. Those specific FA MMQ variants are gated off at the
    # shader-gen and pipeline-registration sites (search for "qvac: FlashAttn
    # MMQ"); the rest of the integer-dot family (mul_mat_mat_q8_1,
    # mul_mat_vec_q8_1, etc.) compiles cleanly and is essential for fast TQ /
    # quant matmul on Mali. So feature-test integer-dot unconditionally — no
    # global -DGGML_VULKAN_DISABLE_INTEGER_DOT opt-out needed any more.
    test_shader_extension_support(
        "GL_EXT_integer_dot_product"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/integer_dot.comp"
        "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
    )

    test_shader_extension_support(
        "GL_EXT_bfloat16"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/bfloat16.comp"
        "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"
    )

    target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)
    if (TARGET SPIRV-Headers::SPIRV-Headers)
        # Provides <spirv/unified1/spirv.hpp> on platforms (e.g. Android NDK)
        # whose Vulkan package does not bundle the SPIRV headers. It is a
        # header-only dependency needed only to compile ggml-vulkan.cpp, so wrap
        # it in BUILD_INTERFACE: as a static lib the PRIVATE link would otherwise
        # leak into the installed ggml-targets export and break downstream
        # find_package(ggml) on platforms whose Vulkan SDK ships the config
        # target (e.g. Windows), which never call find_package(SPIRV-Headers).
        target_link_libraries(ggml-vulkan PRIVATE $<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>)
    endif()
    if (ANDROID AND GGML_VK_CPP_HEADERS_DIR)
        # NDK has no <vulkan/vulkan.hpp>; prepend the fetched C++ headers.
        target_include_directories(ggml-vulkan BEFORE PRIVATE "${GGML_VK_CPP_HEADERS_DIR}")
    endif()
    target_include_directories(ggml-vulkan
        PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}
            ../../../vendor

    )
    if (ANDROID)
        target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_cpp_wrapper/include")
    endif()

    if (FORCE_GGML_VK_PERF_LOGGER)
      add_compile_definitions(FORCE_GGML_VK_PERF_LOGGER)
    endif()
    # Workaround to the "can't dereference invalidated vector iterator" bug in clang-cl debug build
    # Possibly relevant: https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
    if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
    endif()

    if (GGML_VULKAN_CHECK_RESULTS)
        add_compile_definitions(GGML_VULKAN_CHECK_RESULTS)
    endif()

    if (GGML_VULKAN_DEBUG)
        add_compile_definitions(GGML_VULKAN_DEBUG)
    endif()

    if (GGML_VULKAN_BUILD_ADRENO_SHADERS)
        add_compile_definitions(GGML_VULKAN_BUILD_ADRENO_SHADERS)
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_BUILD_ADRENO_SHADERS=ON)
    endif()

    if (GGML_VULKAN_MEMORY_DEBUG)
        add_compile_definitions(GGML_VULKAN_MEMORY_DEBUG)
    endif()

    if (GGML_VULKAN_SHADER_DEBUG_INFO)
        add_compile_definitions(GGML_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_SHADER_DEBUG_INFO=ON)
    endif()

    if (GGML_VULKAN_VALIDATE)
        add_compile_definitions(GGML_VULKAN_VALIDATE)
    endif()

    if (GGML_VULKAN_RUN_TESTS)
        add_compile_definitions(GGML_VULKAN_RUN_TESTS)
    endif()

    # Set up toolchain for host compilation whether cross-compiling or not
    if (CMAKE_CROSSCOMPILING)
        if (GGML_VULKAN_SHADERS_GEN_TOOLCHAIN)
            set(HOST_CMAKE_TOOLCHAIN_FILE ${GGML_VULKAN_SHADERS_GEN_TOOLCHAIN})
        else()
            detect_host_compiler()
            if (NOT HOST_C_COMPILER OR NOT HOST_CXX_COMPILER)
                message(FATAL_ERROR "Host compiler not found")
            else()
                message(STATUS "Host compiler: ${HOST_C_COMPILER} ${HOST_CXX_COMPILER}")
            endif()
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/host-toolchain.cmake.in ${CMAKE_BINARY_DIR}/host-toolchain.cmake @ONLY)
            set(HOST_CMAKE_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/host-toolchain.cmake)
        endif()
    else()
        # For non-cross-compiling, use empty toolchain (use host compiler)
        set(HOST_CMAKE_TOOLCHAIN_FILE "")
    endif()

    include(ExternalProject)

    if (CMAKE_CROSSCOMPILING)
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${HOST_CMAKE_TOOLCHAIN_FILE})
        message(STATUS "vulkan-shaders-gen toolchain file: ${HOST_CMAKE_TOOLCHAIN_FILE}")
    endif()

    ExternalProject_Add(
        vulkan-shaders-gen
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/$<CONFIG>
                   -DCMAKE_INSTALL_BINDIR=.
                   -DCMAKE_BUILD_TYPE=$<CONFIG>
                   ${VULKAN_SHADER_GEN_CMAKE_ARGS}

        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
        BUILD_ALWAYS  TRUE

        # NOTE: When DESTDIR is set using Makefile generators and
        # "make install" triggers the build step, vulkan-shaders-gen
        # would be installed into the DESTDIR prefix, so it is unset
        # to ensure that does not happen.

        INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=DESTDIR
                        ${CMAKE_COMMAND} --install . --config $<CONFIG>
    )

    set (_ggml_vk_host_suffix $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>,.exe,>)
    set (_ggml_vk_genshaders_dir "${CMAKE_BINARY_DIR}/$<CONFIG>")
    set (_ggml_vk_genshaders_cmd "${_ggml_vk_genshaders_dir}/vulkan-shaders-gen${_ggml_vk_host_suffix}")
    set (_ggml_vk_header     "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp")
    set (_ggml_vk_input_dir  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders")
    set (_ggml_vk_output_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv")

    file(GLOB _ggml_vk_shader_files CONFIGURE_DEPENDS "${_ggml_vk_input_dir}/*.comp")

    # Because external projects do not provide source-level tracking,
    # the vulkan-shaders-gen sources need to be explicitly added to
    # ensure that changes will cascade into shader re-generation.

    file(GLOB _ggml_vk_shaders_gen_sources
              CONFIGURE_DEPENDS "${_ggml_vk_input_dir}/*.cpp"
                                "${_ggml_vk_input_dir}/*.h")

    set (_ggml_vk_common_cpp "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders-common.cpp")

    add_custom_command(
        OUTPUT ${_ggml_vk_header} ${_ggml_vk_common_cpp}
        COMMAND ${_ggml_vk_genshaders_cmd}
            --output-dir ${_ggml_vk_output_dir}
            --target-hpp ${_ggml_vk_header}
            --target-common-cpp ${_ggml_vk_common_cpp}
        DEPENDS ${_ggml_vk_shaders_gen_sources}
                vulkan-shaders-gen
        COMMENT "Generate vulkan shaders header and common cpp"
    )
    target_sources(ggml-vulkan PRIVATE ${_ggml_vk_header} ${_ggml_vk_common_cpp})

    foreach (file_full ${_ggml_vk_shader_files})
        get_filename_component(file ${file_full} NAME)
        set (_ggml_vk_target_cpp "${CMAKE_CURRENT_BINARY_DIR}/${file}.cpp")

        add_custom_command(
            OUTPUT  ${_ggml_vk_target_cpp}
            DEPFILE ${_ggml_vk_target_cpp}.d
            COMMAND ${_ggml_vk_genshaders_cmd}
                --glslc      ${Vulkan_GLSLC_EXECUTABLE}
                --source     ${file_full}
                --output-dir ${_ggml_vk_output_dir}
                --target-hpp ${_ggml_vk_header}
                --target-cpp ${_ggml_vk_target_cpp}
            DEPENDS ${file_full}
                    ${_ggml_vk_shaders_gen_sources}
                    vulkan-shaders-gen
            COMMENT "Generate vulkan shaders for ${file}"
        )
        target_sources(ggml-vulkan PRIVATE ${_ggml_vk_target_cpp})
    endforeach()

else()
    message(WARNING "Vulkan not found")
endif()
