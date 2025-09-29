cmake_minimum_required(VERSION 3.11...3.31)

project(vk_denoise_dlssrr LANGUAGES C CXX)

# CMake module FetchContent
include(FetchContent)

#-------------------------------------------------------------------------------
# DLSS dependency

# Tell CMake where to find DLSS_RR SDK, how to get it and where to place it locally.
# CMake will automatically look for CMakeLists.txt in there
FetchContent_Declare(
    DLSS_SDK
    GIT_REPOSITORY https://github.com/NVIDIA/DLSS.git
    GIT_TAG "v310.4.0"
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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

#--------------------------------------------------------------------------------------------------
# Add example
add_subdirectory(dlss_rr)
