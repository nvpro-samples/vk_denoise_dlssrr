# Common default CMake header for nvpro-core samples
cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# CMake module FetchContent
include(FetchContent)

#-------------------------------------------------------------------------------
# DLSS dependency

# Tell CMake where to find DLSS_RR SDK, how to get it and where to place it locally.
# CMake will automatically look for CMakeLists.txt in there
FetchContent_Declare(
    DLSS_SDK
    GIT_REPOSITORY https://github.com/NVIDIA/DLSS.git
    GIT_TAG "v310.2.1"
    GIT_SHALLOW
    GIT_SUBMODULES
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/dlss_sdk
)

# DLSS doesn't come with a CMakeLists.txt
FetchContent_MakeAvailable(
    DLSS_SDK
)

option(NGX_SDK_ROOT "Path to NGX/DLSS_RR SDK" "${CMAKE_CURRENT_SOURCE_DIR}/externals/dlss_sdk/")
set(NGX_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/dlss_sdk/")

include(cmake/ngx.cmake)


project(${PROJECT_NAME} LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#--------------------------------------------------------------------------------------------------
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()

## Various functions and macros REQUIRED
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

_add_project_definitions(${PROJECT_NAME})
set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

if(MSVC)
    add_definitions(/wd26812)  # 'enum class' over 'enum'
    add_definitions(/wd26451)  # Arithmetic overflow, casting 4 byte value to 8 byte value
else()
    add_compile_options("-fno-omit-frame-pointer")
endif()

#--------------------------------------------------------------------------------------------------
# Add example
add_subdirectory(dlss_rr)

#--------------------------------------------------------------------------------------------------
# Install - copying the media directory
message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})
install(DIRECTORY "media" CONFIGURATIONS Release RelWithDebInfo MinSizeRel DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
