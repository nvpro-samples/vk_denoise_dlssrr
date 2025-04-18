#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.h)

_add_package_VulkanSDK()
_add_package_ImGUI()
_add_nvpro_core_lib()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR} ${ADVANCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_DEFINE_MATH_OPERATORS)
source_group("Other"  FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

# Link against DLSS
target_link_libraries(${PROJECT_NAME} ngx)

# Adding libraries
target_link_libraries (${PROJECT_NAME} 
        # All nvpro-core dependencies  
        nvpro_core                       #  "                 " 
        optimized ${LIBRARIES_OPTIMIZED} #  "                 " 
        debug ${LIBRARIES_DEBUG}         #  "                 " 
        ${PLATFORM_LIBRARIES}            #  "                 " 
        )

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Local shaders
file(GLOB SHD_HDR ${SHD_DIR}/*.h ${SHD_DIR}/*.glsl)
file(GLOB SHD_SRC ${SHD_DIR}/*.frag ${SHD_DIR}/*.rgen ${SHD_DIR}/*.rchit ${SHD_DIR}/*.rahit ${SHD_DIR}/*.rmiss ${SHD_DIR}/*.comp)

# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_SRC}
    HEADER_FILES ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
    VULKAN_TARGET "vulkan1.3"
    HEADER ON
    DEPENDENCY ${Vulkan_BUILD_DEPENDENCIES}
    FLAGS -I${SHD_DIR} -I${NVPRO_CORE_DIR}
    )

target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS} "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
source_group("Shaders/src"   FILES ${GLSL_SOURCES} )
source_group("Shaders/hdr"   FILES ${GLSL_HEADERS} )
source_group("Documentation"   FILES "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")

download_files(FILENAMES "ABeautifulGame.zip" EXTRACT)
download_files(FILENAMES "daytime.hdr" "environment.hdr" "hdri-19_color_small.hdr" "std_env.hdr")

# Copy binary
_finalize_target( ${PROJECT_NAME} )

get_target_property(DLSS_LIBS ngx EXTRA_DLLS)

#message(DLSS_LIBS: ${DLSS_LIBS})

# Copy the DLSS_RR libraries to the intermediate build directories as well
add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                            "${DLSS_LIBS}"
                            $<TARGET_FILE_DIR:${PROJECT_NAME}>
                    COMMAND_EXPAND_LISTS)

# Install DLSS_RR libraries as part of the INSTALL build target
install(FILES "${DLSS_LIBS}" CONFIGURATIONS Release RelWithDebInfo MinSizeRel DESTINATION bin_${ARCH})
install(FILES "${DLSS_LIBS}" CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
