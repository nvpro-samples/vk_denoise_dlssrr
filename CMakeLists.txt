cmake_minimum_required(VERSION 3.11...3.31)

project(vk_denoise_dlssrr LANGUAGES C CXX)

#--------------------------------------------------------------------------------------------------
# nvpro_core2 core library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# NGX (DLSS) dependency
find_package(NGX REQUIRED)

#--------------------------------------------------------------------------------------------------
# Add example
add_subdirectory(dlss_rr)
