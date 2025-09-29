/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vulkan/vulkan_core.h>

#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_defs_dlssd.h"

#include <glm/glm.hpp>

#include <array>
#include <filesystem>
#include <vector>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper to convert NGX error codes to a string
std::string getNGXResultString(NVSDK_NGX_Result result);

// Helper to make error code checking easier
NVSDK_NGX_Result checkNgxResult(NVSDK_NGX_Result result, const char* func, int line);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// NGXApiContext encapsulates access to the generic NGX API. DlssRR is just one of multiple NGX technologies;
// implemented as plugins called 'snippets'
class DlssRR;

class NgxContext
{
public:
  NgxContext() = default;
  ~NgxContext();


  struct NgxInitInfo
  {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    uint32_t         queueFamilyIdx = 0;
#ifdef NDEBUG
    NVSDK_NGX_Logging_Level loggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
#else
    NVSDK_NGX_Logging_Level loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
#endif
    std::filesystem::path applicationPath;  // directory to store temporary files and logs in
  };

  // Initialize the NGX context on the given Vulkan device
  NVSDK_NGX_Result init(const NgxInitInfo& initInfo);

  // Do not destroy NgxContext before all instances of DLSS_RR are destroyed.
  void deinit();

  struct SupportedSizes
  {
    VkExtent2D minSize     = {};
    VkExtent2D maxSize     = {};
    VkExtent2D optimalSize = {};
  };
  struct QuerySizeInfo
  {
    VkExtent2D                  outputSize;
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  };
  // Returns the supported input render size size
  NVSDK_NGX_Result querySupportedDlssInputSizes(const QuerySizeInfo& outputSize, SupportedSizes& renderSizes);


  struct DlssRRInitInfo
  {
    VkExtent2D                                     inputSize  = {};  // dimensions of the noisy input textures.
    VkExtent2D                                     outputSize = {};  // dimensions of the output after denoising.
    NVSDK_NGX_PerfQuality_Value                    quality    = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    NVSDK_NGX_RayReconstruction_Hint_Render_Preset preset     = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default;
  };
  // Initialize a DlssRR instance. There can be multiple.
  NVSDK_NGX_Result initDlssRR(const DlssRRInitInfo& initInfo, DlssRR& dlssrr);

  // Check if DLSS_RR is available and createDlssRR() can be called
  static NVSDK_NGX_Result isDlssRRAvailable(VkInstance instance, VkPhysicalDevice physicalDevice);

  // Append 'extensions' with the instance extensions that should be enabled for DLSS_RR
  static NVSDK_NGX_Result getDlssRRRequiredInstanceExtensions(std::vector<VkExtensionProperties>& extensions);

  // Append 'extensions' with the device extensions that should be enabled for DLSS_RR
  static NVSDK_NGX_Result getDlssRRRequiredDeviceExtensions(VkInstance                          instance,
                                                            VkPhysicalDevice                    device,
                                                            std::vector<VkExtensionProperties>& extensions);

private:
  // We don't provide proper operators, so forbid copying & moving for now
  NgxContext(const NgxContext&)             = delete;
  NgxContext(const NgxContext&&)            = delete;
  NgxContext& operator=(const NgxContext&)  = delete;
  NgxContext& operator=(const NgxContext&&) = delete;

  VkDevice             m_device    = VK_NULL_HANDLE;
  VkQueue              m_queue     = VK_NULL_HANDLE;
  NVSDK_NGX_Parameter* m_ngxParams = nullptr;
  std::wstring         m_applicationPath;
  uint32_t             m_queueFamilyIdx = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class DlssRR
{
public:
  DlssRR() = default;
  ~DlssRR();

  void deinit();

  // Names of the various DLSS_RR input and output resources.
  enum DlssResource
  {
    // Mandatory buffers
    RESOURCE_COLOR_IN = 0,
    RESOURCE_COLOR_OUT,
    RESOURCE_DIFFUSE_ALBEDO,
    RESOURCE_SPECULAR_ALBEDO,
    RESOURCE_NORMALROUGHNESS,
    RESOURCE_MOTIONVECTOR,
    RESOURCE_LINEARDEPTH,
    // Below are optional guide buffers
    RESOURCE_SPECULAR_HITDISTANCE,

    RESOURCE_NUM
  };

  // Associate a DlssRR resource with a Vulkan texture
  void setResource(DlssResource resourceId, VkImage image, VkImageView imageView, VkFormat format);
  void resetResource(DlssResource resourceId);

  // Perform the actual denoising.
  // 'renderSize' is the subrectangle ( [0,0] based) of the input textures that has been rendered to.
  // 'jitter' contains the current frame's jitter [-0.5..0.5]
  // 'modelView' and 'projection' define the camera
  // Use 'reset' if the denoiser should discard its history (for instance upon a drastic change in the scene, like cutscenes)
  NVSDK_NGX_Result denoise(VkCommandBuffer  cmd,
                           glm::uvec2       renderSize,
                           glm::vec2        jitter,
                           const glm::mat4& modelView,
                           const glm::mat4& projection,
                           bool             reset = false);


private:
  friend class NgxContext;
  NVSDK_NGX_Result init(VkDevice device, VkQueue queue, uint32_t queueFamilyIdx, NVSDK_NGX_Parameter* ngxParams, const NgxContext::DlssRRInitInfo& info);

  // We don't provide proper operators, so forbid copying & moving for now
  DlssRR(const DlssRR&)             = delete;
  DlssRR(const DlssRR&&)            = delete;
  DlssRR& operator=(const DlssRR&)  = delete;
  DlssRR& operator=(const DlssRR&&) = delete;

  VkDevice                                        m_device      = VK_NULL_HANDLE;
  NVSDK_NGX_Parameter*                            m_ngxParams   = nullptr;
  NVSDK_NGX_Handle*                               m_dlssdHandle = nullptr;
  VkExtent2D                                      m_inputSize;
  VkExtent2D                                      m_outputSize;
  std::array<NVSDK_NGX_Resource_VK, RESOURCE_NUM> m_resources;
};
