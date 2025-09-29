#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------

message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

file(GLOB CPP_SOURCES src/*.cpp src/*.hpp src/*.h)


#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Local shaders - Slang (new)
file(GLOB SHD_SLANG_HDR ${SHD_DIR}/*.h)
file(GLOB SHD_SLANG_SRC ${SHD_DIR}/*.slang)

# Shaders to the build
list(APPEND SHADER_SLANG_FILES
    ${NVSHADERS_DIR}/nvshaders/sky_physical.slang
    ${NVSHADERS_DIR}/nvshaders/sky_simple.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_diffuse.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_glossy.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_dome.slang
    ${NVSHADERS_DIR}/nvshaders/hdr_integrate_brdf.slang
    ${NVSHADERS_DIR}/nvshaders/tonemapper.slang

    ${SHD_DIR}/primary_rgen.slang
    ${SHD_DIR}/primary_rmiss.slang
    ${SHD_DIR}/primary_rchit.slang
    ${SHD_DIR}/secondary_rchit.slang
    ${SHD_DIR}/secondary_rahit.slang
    ${SHD_DIR}/secondary_rmiss.slang
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/_autogen")

compile_slang(
    "${SHADER_SLANG_FILES}"
    "${SHADER_OUTPUT_DIR}"
    GENERATED_SHADER_HEADERS
    EXTRA_FLAGS "-I${NVSHADERS_DIR}"
)

message(STATUS "NVSHADERS_DIR ${NVSHADERS_DIR}")

source_group("Shaders" FILES  ${SHADER_SLANG_FILES} ${SHD_SLANG_HDR})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_HEADERS})
source_group("Documentation" FILES "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")

add_executable(${PROJECT_NAME} ${CPP_SOURCES})

target_sources(${PROJECT_NAME} PRIVATE ${SHD_SLANG_SRC} ${SHD_SLANG_HDR} ${GENERATED_SHADER_HEADERS} "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
# Let it find the "./_autogen/" shaders
target_include_directories(${PROJECT_NAME} PRIVATE ${SHADER_OUTPUT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvshaders_host
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvgui
  nvpro2::nvvkgltf
  nvpro2::nvaftermath
  ngx
  tinygltf
)


download_files(FILENAMES "ABeautifulGame.zip" EXTRACT)
download_files(FILENAMES "daytime.hdr" "environment.hdr" "hdri-19_color_small.hdr" "std_env.hdr")


add_project_definitions(${PROJECT_NAME})


get_target_property(DLSS_LIBS ngx EXTRA_DLLS)
# Copy the DLSS_RR libraries to the intermediate build directories as well
add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                            "${DLSS_LIBS}"
                            $<TARGET_FILE_DIR:${PROJECT_NAME}>
                    COMMAND_EXPAND_LISTS)

# Install DLSS_RR libraries as part of the INSTALL build target
install(FILES "${DLSS_LIBS}" CONFIGURATIONS Release RelWithDebInfo MinSizeRel DESTINATION bin_${ARCH})
install(FILES "${DLSS_LIBS}" CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)

# Make Visual Studio use this project as the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
