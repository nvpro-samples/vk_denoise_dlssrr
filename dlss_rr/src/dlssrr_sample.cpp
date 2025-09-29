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

//////////////////////////////////////////////////////////////////////////
/*

 This sample loads GLTF scenes and renders them using RTX (path tracer)

 The path tracer renders into multiple G-Buffers, which are used
 to denoise the image using DLSS_RR.
 */
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vulkan/vulkan_core.h>

#define IMGUI_DEFINE_MATH_OPERATORS

#define VMA_IMPLEMENTATION
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include "nvapp/application.hpp"

#include "nvvk/ray_picker.hpp"
#include "nvvk/sbt_generator.hpp"

#include "nvapp/elem_camera.hpp"
#include "nvapp/elem_dbgprintf.hpp"
#include "nvapp/elem_default_title.hpp"
#include "nvapp/elem_default_menu.hpp"
#include "nvapp/elem_logger.hpp"

#include "nvgui/file_dialog.hpp"
#include "nvgui/property_editor.hpp"
#include "nvgui/sky.hpp"
#include "nvgui/camera.hpp"
#include "nvgui/tonemapper.hpp"

#include "nvutils/logger.hpp"
#include "nvutils/file_operations.hpp"
#include "nvutils/camera_manipulator.hpp"

#include "nvvk/barriers.hpp"
#include "nvvk/gbuffers.hpp"
#include "nvvk/context.hpp"
#include "nvvk/descriptors.hpp"
#include "nvvk/shaders.hpp"
#include "nvvk/validation_settings.hpp"

#include "nvvkgltf/scene_rtx.hpp"
#include "nvvkgltf/scene_vk.hpp"

#include "nvvk/hdr_ibl.hpp"
#include "nvvk/pipeline.hpp"

#include "nvshaders_host/sky.hpp"
#include "nvshaders_host/tonemapper.hpp"

#include "primary_rgen.slang.h"
#include "primary_rchit.slang.h"
#include "primary_rmiss.slang.h"
#include "secondary_rahit.slang.h"
#include "secondary_rchit.slang.h"
#include "secondary_rmiss.slang.h"

#include "tonemapper.slang.h"
#include "sky_physical.slang.h"
#include "hdr_dome.slang.h"

#include "shaders/host_device.h"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/sky_io.h.slang"

#include "dlssrr_wrapper.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#include <array>
#include <filesystem>
#include <math.h>
#include <memory>

using namespace glm;

std::shared_ptr<nvapp::ElementCamera>    g_elem_camera;
std::shared_ptr<nvapp::ElementDbgPrintf> g_dbgPrintf;

// Little desparate helper to allo me set a breakpoint on that exit()
void myExit()
{
  exit(EXIT_FAILURE);
}

#define NGX_ABORT_ON_FAIL(x)                                                                                           \
  {                                                                                                                    \
    NVSDK_NGX_Result result = checkNgxResult((x), __func__, __LINE__);                                                 \
    if(NVSDK_NGX_FAILED(result))                                                                                       \
      myExit();                                                                                                        \
  }

#define NGX_CHECK(x) checkNgxResult((x), __func__, __LINE__);


// #DLSS_RR
// halton low discrepancy sequence, from https://www.shadertoy.com/view/wdXSW8
vec2 halton(int index)
{
  const vec2 coprimes = vec2(2.0F, 3.0F);
  vec2       s        = vec2(index, index);
  vec4       a        = vec4(1, 1, 0, 0);
  while(s.x > 0. && s.y > 0.)
  {
    a.x = a.x / coprimes.x;
    a.y = a.y / coprimes.y;
    a.z += a.x * fmod(s.x, coprimes.x);
    a.w += a.y * fmod(s.y, coprimes.y);
    s.x = floorf(s.x / coprimes.x);
    s.y = floorf(s.y / coprimes.y);
  }
  return vec2(a.z, a.w);
}

// Main sample class
class DlssApplet : public nvapp::IAppElement
{
  enum RenderBufferName
  {
    eGBufBaseColor_Metalness,
    eGBufSpecAlbedo,
    eGBufSpecHitDist,
    eGBufNormalRoughness,
    eGBufMotionVectors,
    eGBufViewZ,
    eGBufColor,
    eNumRenderBufferNames
  };

  enum OutputBufferName
  {
    eGBufColorOut,  // denoised
    eGBufLdr,

    eNumOutputBufferNames
  };

  struct Settings
  {
    int       maxFrames{200000};
    int       maxDepth{5};
    glm::vec4 envIntensity{1.F};
    float     envRotation{0.F};
  } m_settings;

public:
  DlssApplet()           = default;
  ~DlssApplet() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice         = app->getPhysicalDevice();
    allocator_info.device                 = app->getDevice();
    allocator_info.instance               = app->getInstance();
    allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    //FIXME: no way for onAttach to return failure
    NVVK_CHECK(m_alloc.init(allocator_info));  // Allocator


    m_stagingUploader.init(&m_alloc);  // void
    m_stagingUploader.setEnableLayoutBarriers(true);

    m_samplerPool.init(m_device);  // void

    m_sceneVk.init(&m_alloc, &m_samplerPool);  // GLTF Scene buffers
    m_sceneRtx.init(&m_alloc);  //void                                                               // GLTF Scene BLAS/TLAS

    m_tonemapper.init(&m_alloc, tonemapper_slang);  // void
    m_picker.init(&m_alloc);

    m_skyEnv.init(&m_alloc, sky_physical_slang);  //void
    m_alloc.createBuffer(m_skyParamBuffer, sizeof(shaderio::SkyPhysicalParameters), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT);

    m_hdrEnv.init(&m_alloc, &m_samplerPool);  //void


    // Requesting ray tracing properties (this can be moved into m_sbt.init()
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rt_prop;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create the Shading Binding Table (SBT)
    uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.init(m_app->getDevice(), rt_prop);  // void

    m_outputSize = {app->getWindowSize().width, app->getWindowSize().height};

    createVulkanBuffers();

    // #DLSS
    {
      if(NVSDK_NGX_FAILED(NgxContext::isDlssRRAvailable(m_app->getInstance(), m_app->getPhysicalDevice())))
      {
        LOGE("DLSS is not available, aborting.\n");
        exit(EXIT_FAILURE);
        return;
      }

      NGX_ABORT_ON_FAIL(m_ngx.init({.instance        = m_app->getInstance(),
                                    .physicalDevice  = m_app->getPhysicalDevice(),
                                    .device          = m_app->getDevice(),
                                    .queue           = m_app->getQueue(0).queue,
                                    .applicationPath = nvutils::getExecutablePath().parent_path()}));

      m_dlssBufferEnable.fill(true);
    }
    createDlssSet();

    // Create resources in DLSS_RR input render size and output size
    createInputGbuffers(m_renderSize);
    createOutputGbuffer(m_outputSize);

    m_cameraManip = std::make_shared<nvutils::CameraManipulator>();
    g_elem_camera->setCameraManipulator(m_cameraManip);
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void reinitDlss(bool querySizes)
  {
    vkDeviceWaitIdle(m_device);

    m_dlss.deinit();

    if(querySizes)
    {
      NGX_ABORT_ON_FAIL(m_ngx.querySupportedDlssInputSizes({.outputSize = {m_outputSize.x, m_outputSize.y}, .quality = m_dlssQuality},
                                                           m_dlssSizes));
      m_renderSize = {m_dlssSizes.optimalSize.width, m_dlssSizes.optimalSize.height};
    }

    {
      std::vector<VkExtensionProperties> extensions;
      m_ngx.getDlssRRRequiredInstanceExtensions(extensions);

      for(auto& e : extensions)
      {
        std::cout << e.extensionName << std::endl;
      }
    }

    {
      std::vector<VkExtensionProperties> extensions;
      m_ngx.getDlssRRRequiredDeviceExtensions(m_app->getInstance(), m_app->getPhysicalDevice(), extensions);
      std::cout << "Device Extensions " << std::endl;
      for(auto& e : extensions)
      {
        std::cout << e.extensionName << std::endl;
      }
    }

    NGX_ABORT_ON_FAIL(m_ngx.initDlssRR({.inputSize  = {m_renderSize.x, m_renderSize.y},
                                        .outputSize = {m_outputSize.x, m_outputSize.y},
                                        .quality    = m_dlssQuality,
                                        .preset     = m_dlssPreset},
                                       m_dlss));

    createInputGbuffers(m_renderSize);
  }

  void setDlssResources()
  {
    auto dlssRenderResourceFromGBufTexture = [&](DlssRR::DlssResource dlssResource, RenderBufferName gbufIndex) {
      m_dlssBufferEnable[gbufIndex] ? m_dlss.setResource(dlssResource, m_renderBuffers.getColorImage(gbufIndex),
                                                         m_renderBuffers.getDescriptorImageInfo(gbufIndex).imageView,
                                                         m_renderBuffers.getColorFormat(gbufIndex)) :
                                      m_dlss.resetResource(dlssResource);
    };

    // #DLSS provide the input and guide buffers to DLSS_RR
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_COLOR_IN, eGBufColor);
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_NORMALROUGHNESS, eGBufNormalRoughness);
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_MOTIONVECTOR, eGBufMotionVectors);
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_LINEARDEPTH, eGBufViewZ);
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_DIFFUSE_ALBEDO, eGBufBaseColor_Metalness);
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_SPECULAR_ALBEDO, eGBufSpecAlbedo);
    dlssRenderResourceFromGBufTexture(DlssRR::RESOURCE_SPECULAR_HITDISTANCE, eGBufSpecHitDist);

    auto dlssOutputResourceFromGBufTexture = [&](DlssRR::DlssResource dlssResource, OutputBufferName gbufIndex) {
      m_dlss.setResource(dlssResource, m_outputBuffers.getColorImage(gbufIndex),
                         m_outputBuffers.getDescriptorImageInfo(gbufIndex).imageView, m_outputBuffers.getColorFormat(gbufIndex));
    };
    dlssOutputResourceFromGBufTexture(DlssRR::RESOURCE_COLOR_OUT, eGBufColorOut);
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    vkDeviceWaitIdle(m_device);

    m_outputSize = {size.width, size.height};
    // #DLSS
    // Work around a bug in DLSS_RR that causes a crash below a certain image size
    m_outputSize = glm::max({256, 256}, m_outputSize);

    createOutputGbuffer(m_outputSize);
    reinitDlss(true);
  }

  void onUIMenu() override
  {
    bool load_file{false};

    windowTitle();

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Load", "Ctrl+O"))
      {
        load_file = true;
      }
      ImGui::Separator();
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      load_file = true;
    }

    if(load_file)
    {
      auto filename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                  "glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
      onFileDrop(filename.c_str());
    }
  }

  void onFileDrop(const std::filesystem::path& filename) override
  {
    namespace fs = std::filesystem;

    // Make sure none of the resources is still in use
    vkDeviceWaitIdle(m_device);

    auto extension = filename.extension();
    if(extension == fs::path(".gltf") || extension == fs::path(".glb"))
    {
      createScene(filename);
    }
    else if(extension == ".hdr")
    {
      createHdr(filename);
      resetFrame();
    }

    resetFrame();
  }

  void onUIRender() override
  {
    using namespace nvgui;

    bool reset{false};
    // Pick under mouse cursor
    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
    {
      screenPicking();
    }

    {  // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::CollapsingHeader("Camera"))
      {
        CameraWidget(m_cameraManip);
      }

      if(ImGui::CollapsingHeader("Settings"))
      {
        PropertyEditor::begin();

        if(PropertyEditor::treeNode("Ray Tracing"))
        {
          reset |= PropertyEditor::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_settings.maxDepth, 1, 10); });
          reset |= PropertyEditor::entry("Frames",
                                         [&] { return ImGui::DragInt("#3", &m_settings.maxFrames, 5.0F, 1, 1000000); });
          ImGui::SliderFloat("Override Roughness", &m_pushConst.overrideRoughness, 0, 1, "%.3f");
          ImGui::SliderFloat("Override Metalness", &m_pushConst.overrideMetallic, 0, 1, "%.3f");

          PropertyEditor::treePop();
        }
        bool flipBitangent = m_pushConst.bitangentFlip < 0 ? true : false;
        PropertyEditor::entry("Flip Bitangent", [&] { return ImGui::Checkbox("##5", &flipBitangent); });
        m_pushConst.bitangentFlip = flipBitangent ? -1.0f : 1.0f;

        bool usePSR = !!(m_frameInfo.flags & FLAGS_USE_PSR);
        PropertyEditor::entry("Use PSR", [&] { return ImGui::Checkbox("##6", &usePSR); }, "Use Primary Surface Replacement on mirrors");
        m_frameInfo.flags = (m_frameInfo.flags & ~FLAGS_USE_PSR) | (usePSR ? FLAGS_USE_PSR : 0);


        bool useRegularization = !!(m_frameInfo.flags & FLAGS_USE_PATH_REGULARIZATION);
        PropertyEditor::entry(
            "Use Path Regularization", [&] { return ImGui::Checkbox("##7", &useRegularization); },
            "Use max. roughness propagation to improve indirect specular highlights");
        m_frameInfo.flags = (m_frameInfo.flags & ~FLAGS_USE_PATH_REGULARIZATION)
                            | (useRegularization ? FLAGS_USE_PATH_REGULARIZATION : 0);

        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Environment"))
      {
        int useSky = m_frameInfo.flags & FLAGS_ENVMAP_SKY;
        reset |= ImGui::RadioButton("Sky", &useSky, FLAGS_ENVMAP_SKY);
        ImGui::SameLine();
        reset |= ImGui::RadioButton("Hdr", &useSky, 0);
        m_frameInfo.flags = (m_frameInfo.flags & ~FLAGS_ENVMAP_SKY) | useSky;

        PropertyEditor::begin();
        PropertyEditor::entry(
            "Intensity",
            [&] {
              static float intensity = 1.0f;
              bool hit = ImGui::SliderFloat("##Color", &intensity, 0, 100, "%.3f", ImGuiSliderFlags_Logarithmic);
              m_settings.envIntensity = glm::vec4(intensity, intensity, intensity, 1);
              return hit;
            },
            "HDR multiplier");

        if(!(m_frameInfo.flags & FLAGS_ENVMAP_SKY))
        {
          PropertyEditor::entry("Rotation", [&] { return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
        }
        else
        {
          nvgui::skyPhysicalParameterUI(m_skyParams);
        }

        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        nvgui::tonemapperWidget(m_tonemapperData);
      }

      if(ImGui::CollapsingHeader("DLSS RR", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        {
          {  // Note that UltraQuality is deliberately left out as unsupported, see DLSS_RR Integration Guide
            const char* const items[] = {"MaxPerf", "Balanced", "MaxQuality", "UltraPerformance", "DLAA"};
            const NVSDK_NGX_PerfQuality_Value itemValues[]{NVSDK_NGX_PerfQuality_Value_MaxPerf, NVSDK_NGX_PerfQuality_Value_Balanced,
                                                           NVSDK_NGX_PerfQuality_Value_MaxQuality,
                                                           NVSDK_NGX_PerfQuality_Value_UltraPerformance,
                                                           NVSDK_NGX_PerfQuality_Value_DLAA};
            // Find item corresponding to currently selected quality
            int item;
            for(item = 0; item < IM_ARRAYSIZE(items) && itemValues[item] != m_dlssQuality; ++item)
              ;
            if(PropertyEditor::entry("Quality", [&]() {
                 return ImGui::ListBox("Quality", &item, items, IM_ARRAYSIZE(items), 3 /*heightInItems*/);
               }))
            {
              m_dlssQuality = itemValues[item];
              reinitDlss(true);
              reset = true;
            }
          }

          {  // Some of the presets are marked as "Do not use". See nvsdk_ngx_defs.h
            const char* const                              items[]      = {"Default", "Preset D", "Preset E"};
            NVSDK_NGX_RayReconstruction_Hint_Render_Preset itemValues[] = {
                NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default,  // default behavior, may or may not change after OTA
                NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D, NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E};

            // Find item corresponding to currently selected preset
            int item;
            for(item = 0; item < IM_ARRAYSIZE(items) && itemValues[item] != m_dlssPreset; ++item)
              ;
            if(PropertyEditor::entry("Presets", [&]() {
                 return ImGui::ListBox("Presets", &item, items, IM_ARRAYSIZE(items), 3 /*heightInItems*/);
               }))
            {
              m_dlssPreset = itemValues[item];
              reinitDlss(true);
              reset = true;
            }
          }

          {
            bool  renderResolutionChange = false;
            int   width                  = (int)m_renderSize.x;
            int   height                 = (int)m_renderSize.y;
            float aspect                 = (float)m_dlssSizes.optimalSize.width / (float)m_dlssSizes.optimalSize.height;

            auto slider = [&](int& value, int minValue, int maxValue) {
              ImGui::Text("%d", minValue);
              ImGui::SameLine();
              bool changed = ImGui::SliderInt("##", &value, minValue, maxValue, "%d", ImGuiSliderFlags_AlwaysClamp);
              ImGui::SameLine();
              ImGui::Text("%u", maxValue);
              return changed;
            };

            bool widthChanged = PropertyEditor::entry(
                "Input Width",
                [&]() { return slider(width, (int)m_dlssSizes.minSize.width, (int)m_dlssSizes.maxSize.width); },
                "Size of the DLSS_RR input buffers");

            if(widthChanged)
            {
              height                 = int((float)width / aspect);
              renderResolutionChange = true;
            }

            bool heightChanged = PropertyEditor::entry(
                "Input Height",
                [&]() { return slider(height, (int)m_dlssSizes.minSize.height, (int)m_dlssSizes.maxSize.height); },
                "Size of the DLSS_RR input buffers");

            if(heightChanged)

            {
              width                  = (int)((float)height * aspect);
              renderResolutionChange = true;
            }

            if(renderResolutionChange)
            {
              m_renderSize.x = width;
              m_renderSize.y = height;

              reinitDlss(false);
              reset = true;
            }
          }

          PropertyEditor::entry(
              "Show Buffers Scaled", [&]() { return ImGui::Checkbox("##", &m_dlssShowScaledBuffers); },
              "Whether to show the input at their native resolution or scaled to the viewport");
        }
        PropertyEditor::end();

        ImVec2 tumbnailSize = {100 * m_renderBuffers.getAspectRatio(), 100};

        auto showBuffer = [&](const char* name, RenderBufferName buffer, bool optional = false) {
          ImGui::PushID(name);
          ImGui::TableNextColumn();
          if(ImGui::ImageButton(name, (ImTextureID)m_renderBuffers.getDescriptorSet(buffer), tumbnailSize))
          {
            m_showBuffer = buffer;
          }
          if(optional)
          {
            ImGui::Checkbox("##enable", &m_dlssBufferEnable[buffer]);
            ImGui::SameLine();
          }
          ImGui::Text("%s", name);
          ImGui::PopID();
        };

        if(ImGui::BeginTable("Thumbnails", 2))
        {

          ImGui::Text("Guide Buffers");
          ImGui::TableNextRow();
          showBuffer("Color", eGBufColor);
          showBuffer("Diffuse Albedo", eGBufBaseColor_Metalness);
          ImGui::TableNextRow();
          showBuffer("Specular Albedo", eGBufSpecAlbedo);
          showBuffer("Normal/Roughness", eGBufNormalRoughness);
          ImGui::TableNextRow();
          showBuffer("Motion vectors", eGBufMotionVectors);
          showBuffer("ViewZ", eGBufViewZ);
          ImGui::TableNextRow();
          showBuffer("Specular Hitdist", eGBufSpecHitDist, true);

          ImGui::TableNextColumn();
          ImGui::TableNextColumn();

          ImGui::Text("Denoised & Tonemapped Output");
          if(ImGui::ImageButton("Denoised", (ImTextureID)m_outputBuffers.getDescriptorSet(eGBufLdr), tumbnailSize))
          {
            m_showBuffer = eNumRenderBufferNames;
          }

          ImGui::EndTable();
        }
      }

      ImGui::End();

      if(reset)
      {
        resetFrame();
      }
    }

    {
      // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      ImVec2 imageSize = m_dlssShowScaledBuffers ? ImGui::GetContentRegionAvail() :
                                                   ImVec2(float(m_renderSize.x), float(m_renderSize.y));
      // Display the G-Buffer image in the main viewport
      (m_showBuffer == eNumRenderBufferNames) ?
          ImGui::Image((ImTextureID)m_outputBuffers.getDescriptorSet(eGBufLdr), ImGui::GetContentRegionAvail()) :
          ImGui::Image((ImTextureID)m_renderBuffers.getDescriptorSet(m_showBuffer), imageSize);

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_scene.valid())
    {
      return;
    }

    NVVK_DBG_SCOPE(cmd);

    // Get camera info
    float view_aspect_ratio = (float)m_outputSize.x / m_outputSize.y;

    m_frameInfo.prevMVP = m_frameInfo.proj * m_frameInfo.view;

    // Update Frame buffer uniform buffer
    const auto& clip = m_cameraManip->getClipPlanes();
    m_frameInfo.view = m_cameraManip->getViewMatrix();
    m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(m_cameraManip->getFov()), view_aspect_ratio, clip.x, clip.y);

    // Were're feeding the raytracer with a flipped matrix for convenience
    m_frameInfo.proj[1][1] *= -1;

    m_frameInfo.projInv      = glm::inverse(m_frameInfo.proj);
    m_frameInfo.viewInv      = glm::inverse(m_frameInfo.view);
    m_frameInfo.envRotation  = m_settings.envRotation;
    m_frameInfo.envIntensity = m_settings.envIntensity;
    m_frameInfo.jitter       = halton(m_frame) - vec2(0.5);

    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &m_frameInfo);

    // Push constant
    m_pushConst.maxDepth   = m_settings.maxDepth;
    m_pushConst.frame      = m_frame;
    m_pushConst.mouseCoord = g_dbgPrintf->getMouseCoord();

    // Helper lambdas to make writing image pipeline barriers easier
    auto imageShaderWriteToRead = [](VkImage image, VkPipelineStageFlagBits2 srcStage, VkPipelineStageFlagBits2 dstStage) {
      return nvvk::makeImageMemoryBarrier({
          .image         = image,
          .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
          .srcStageMask  = srcStage,
          .dstStageMask  = dstStage,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      });
    };
    auto imageShaderReadToWrite = [](VkImage image, VkPipelineStageFlagBits2 srcStage, VkPipelineStageFlagBits2 dstStage) {
      return nvvk::makeImageMemoryBarrier({.image         = image,
                                           .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
                                           .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
                                           .srcStageMask  = srcStage,
                                           .dstStageMask  = dstStage,
                                           .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                           .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT});
    };

    auto gbufferShaderWriteToRead = [&]<typename T, size_t N, typename G>(const G& gbuffer, const T(&buffers)[N],
                                                                          VkPipelineStageFlagBits2 srcStage,
                                                                          VkPipelineStageFlagBits2 dstStage) {
      std::array<VkImageMemoryBarrier2, N> x;
      for(size_t i = 0; i < N; ++i)
        x[i] = imageShaderWriteToRead(gbuffer.getColorImage(buffers[i]), srcStage, dstStage);
      return x;
    };
    auto gbufferShaderReadToWrite = [&]<typename T, size_t N, typename G>(const G& gbuffer, const T(&buffers)[N],
                                                                          VkPipelineStageFlagBits2 srcStage,
                                                                          VkPipelineStageFlagBits2 dstStage) {
      std::array<VkImageMemoryBarrier2, N> x;
      for(size_t i = 0; i < N; ++i)
        x[i] = imageShaderReadToWrite(gbuffer.getColorImage(buffers[i]), srcStage, dstStage);
      return x;
    };

    auto renderBufferShaderWriteToRead = [&]<std::size_t N>(const RenderBufferName(&buffers)[N], VkPipelineStageFlagBits2 srcStage,
                                                            VkPipelineStageFlagBits2 dstStage) {
      return gbufferShaderWriteToRead(m_renderBuffers, buffers, srcStage, dstStage);
    };
    auto renderBufferShaderReadToWrite = [&]<std::size_t N>(const RenderBufferName(&buffers)[N], VkPipelineStageFlagBits2 srcStage,
                                                            VkPipelineStageFlagBits2 dstStage) {
      return gbufferShaderReadToWrite(m_renderBuffers, buffers, srcStage, dstStage);
    };
    auto outputBufferShaderReadToWrite = [&]<std::size_t N>(const OutputBufferName(&buffers)[N], VkPipelineStageFlagBits2 srcStage,
                                                            VkPipelineStageFlagBits2 dstStage) {
      return gbufferShaderReadToWrite(m_outputBuffers, buffers, srcStage, dstStage);
    };
    auto outputBufferShaderWriteToRead = [&]<std::size_t N>(const OutputBufferName(&buffers)[N], VkPipelineStageFlagBits2 srcStage,
                                                            VkPipelineStageFlagBits2 dstStage) {
      return gbufferShaderWriteToRead(m_outputBuffers, buffers, srcStage, dstStage);
    };

    auto cmdImageBarriers = [&](const std::initializer_list<const std::span<const VkImageMemoryBarrier2>>& barriers) {
      std::vector<VkImageMemoryBarrier2> final;
      for(auto b : barriers)
        final.insert(final.end(), b.begin(), b.end());

      const VkDependencyInfo depInfo{.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                     .imageMemoryBarrierCount = (uint32_t) final.size(),
                                     .pImageMemoryBarriers    = final.data()};
      vkCmdPipelineBarrier2(cmd, &depInfo);
    };

    // Make Guide Buffers writeable to raytracer
    cmdImageBarriers({renderBufferShaderReadToWrite(
        {eGBufBaseColor_Metalness, eGBufSpecAlbedo, eGBufSpecHitDist, eGBufNormalRoughness, eGBufMotionVectors, eGBufViewZ, eGBufColor},
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)});

    // Pathtrace the scene
    raytraceScene(cmd);

    // Make Guide Buffers readable to DLSS_RR
    cmdImageBarriers({renderBufferShaderWriteToRead({eGBufBaseColor_Metalness, eGBufSpecAlbedo, eGBufSpecHitDist,
                                                     eGBufNormalRoughness, eGBufMotionVectors, eGBufViewZ, eGBufColor},
                                                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
                      outputBufferShaderReadToWrite({eGBufColorOut}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)});

    // #DLSS
    setDlssResources();
    // Check, but don't exit here, because we can disable non-optional guide buffers
    NGX_CHECK(m_dlss.denoise(cmd, m_renderSize, m_frameInfo.jitter, m_frameInfo.view, m_frameInfo.proj, m_frame == 0));

    // Make denoised image readable to tonemapper
    cmdImageBarriers(
        {outputBufferShaderWriteToRead({eGBufColorOut}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
         outputBufferShaderReadToWrite({eGBufLdr}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)});

    // Apply tonemapper
    m_tonemapper.runCompute(cmd, m_outputBuffers.getSize(), m_tonemapperData, m_outputBuffers.getDescriptorImageInfo(eGBufColorOut),
                            m_outputBuffers.getDescriptorImageInfo(eGBufLdr));

    // Make tonemapped image readabble to ImGUI
    cmdImageBarriers({outputBufferShaderReadToWrite({eGBufLdr}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)});

    m_frame++;
  }

private:
  void createScene(const std::filesystem::path& filename)
  {
    m_sceneRtx.destroy();
    m_sceneVk.destroy();
    m_scene.destroy();

    if(!m_scene.load(filename))
    {
      LOGE("Error loading scene");
      return;
    }

    m_cameraManip->fit(m_scene.getSceneBounds().min(), m_scene.getSceneBounds().max());  // Navigation help

    auto cmd = m_app->createTempCmdBuffer();

    {  // Create the Vulkan side of the scene
      m_sceneVk.create(cmd, m_stagingUploader, m_scene);
      m_stagingUploader.cmdUploadAppended(cmd);  //make sure the scene buffers are on the GPU by the time we build
                                                 //the Acceleration Structures
      m_sceneRtx.create(cmd, m_stagingUploader, m_scene, m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);  // Create BLAS / TLAS
      m_stagingUploader.cmdUploadAppended(cmd);
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Descriptor Set and Pipelines
    createSceneSet();
    createRtxSet();
    createRtxPipeline();  // must recreate due to texture changes
    writeSceneSet();
    writeRtxSet();
  }

  void createInputGbuffers(const glm::uvec2& inputSize)
  {
    // Creation of the GBuffers
    m_renderBuffers.deinit();

    VkExtent2D vk_size{inputSize.x, inputSize.y};

    std::vector<VkFormat> colorBuffers(eNumRenderBufferNames);
    // #DLSS
    colorBuffers[eGBufBaseColor_Metalness] = VK_FORMAT_R8G8B8A8_UNORM;
    colorBuffers[eGBufSpecAlbedo]          = VK_FORMAT_R8G8B8A8_UNORM;
    colorBuffers[eGBufSpecHitDist]         = VK_FORMAT_R16_SFLOAT;
    colorBuffers[eGBufNormalRoughness]     = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorBuffers[eGBufMotionVectors]       = VK_FORMAT_R16G16_SFLOAT;
    colorBuffers[eGBufViewZ]               = VK_FORMAT_R16_SFLOAT;
    colorBuffers[eGBufColor]               = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkSampler sampler;
    m_samplerPool.acquireSampler(sampler);

    nvvk::GBufferInitInfo gbInfo = {.allocator      = &m_alloc,
                                    .colorFormats   = colorBuffers,
                                    .imageSampler   = sampler,
                                    .descriptorPool = m_app->getTextureDescriptorPool()};

    m_renderBuffers.init(gbInfo);

    auto cmd = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_renderBuffers.update(cmd, vk_size));
    m_app->submitAndWaitTempCmdBuffer(cmd);

    writeDlssSet();

    // Indicate the renderer to reset its frame
    resetFrame();
  }

  void createOutputGbuffer(const glm::uvec2& outputSize)
  {
    m_outputBuffers.deinit();

    VkExtent2D vk_size{outputSize.x, outputSize.y};

    std::vector<VkFormat> colorBuffers((size_t(eNumOutputBufferNames)));
    colorBuffers[eGBufLdr] = VK_FORMAT_R8G8B8A8_UNORM;

    // #DLSS
    colorBuffers[eGBufColorOut] = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkSampler sampler;
    m_samplerPool.acquireSampler(sampler);

    nvvk::GBufferInitInfo gbInfo = {.allocator      = &m_alloc,
                                    .colorFormats   = colorBuffers,
                                    .imageSampler   = sampler,
                                    .descriptorPool = m_app->getTextureDescriptorPool()};

    m_outputBuffers.init(gbInfo);

    auto cmd = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_outputBuffers.update(cmd, vk_size));
    m_app->submitAndWaitTempCmdBuffer(cmd);

    resetFrame();
  }

  // Create all Vulkan buffer data
  void createVulkanBuffers()
  {
    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    m_rtPipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
    m_rtPipelineLayout = VK_NULL_HANDLE;
    m_alloc.destroyBuffer(m_sbtBuffer);

    // Creating all shaders
    enum StageIndices
    {
      ePrimaryRaygen,
      ePrimaryClosestHit,
      ePrimaryMiss,
      eSecondaryMiss,
      eSecondaryClosestHit,
      eSecondaryAnyHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point

    // #Raygen
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, primary_rgen_slang));
    stage.stage            = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[ePrimaryRaygen] = stage;

    // Miss
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, secondary_rmiss_slang));
    stage.stage            = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eSecondaryMiss] = stage;

    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, primary_rmiss_slang));
    stage.stage          = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[ePrimaryMiss] = stage;

    // AnyHit
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, secondary_rahit_slang));
    stage.stage              = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eSecondaryAnyHit] = stage;

    // Hit Group - Closest Hit
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, secondary_rchit_slang));
    stage.stage                  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eSecondaryClosestHit] = stage;

    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, primary_rchit_slang));
    stage.stage                = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[ePrimaryClosestHit] = stage;

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = ePrimaryRaygen;
    shaderGroups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = ePrimaryMiss;
    shaderGroups.push_back(group);
    group.generalShader = eSecondaryMiss;
    shaderGroups.push_back(group);

    // Primary closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = ePrimaryClosestHit;
    group.anyHitShader     = eSecondaryAnyHit;
    shaderGroups.push_back(group);

    // Secondary closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eSecondaryClosestHit;
    group.anyHitShader     = eSecondaryAnyHit;
    shaderGroups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtxPushConstant)};

    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_rtPipelineLayout,
                                          {m_rtBindings.getLayout(), m_sceneBindings.getLayout(),
                                           m_DlssRRBindings.getLayout(), m_hdrEnv.getDescriptorSetLayout()},
                                          {push_constant}));
    NVVK_DBG_NAME(m_rtPipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    ray_pipeline_info.pGroups                      = shaderGroups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
    ray_pipeline_info.layout                       = m_rtPipelineLayout;

    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &m_rtPipeline);
    NVVK_DBG_NAME(m_rtPipeline);

    // Creating the SBT
    auto sbtSize = m_sbt.calculateSBTBufferSize(m_rtPipeline, ray_pipeline_info);
    m_alloc.createBuffer(m_sbtBuffer, sbtSize, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
                         VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                         m_sbt.getBufferAlignment());
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    m_sbt.populateSBTBuffer(m_sbtBuffer.address, sbtSize, m_sbtBuffer.mapping);

    // Removing temp modules
    for(auto& s : stages)
    {
      vkDestroyShaderModule(m_device, s.module, nullptr);
    }
  }

  void createDlssSet()
  {
    m_DlssRRBindings.deinit();
    nvvk::DescriptorBindings d;

    // #DLSS_RR
    d.addBinding(shaderio::DlssBindings::eNormal_Roughness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(shaderio::DlssBindings::eBaseColor_Metalness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(shaderio::DlssBindings::eSpecAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(shaderio::DlssBindings::eSpecHitDist, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(shaderio::DlssBindings::eViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(shaderio::DlssBindings::eMotionVectors, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d.addBinding(shaderio::DlssBindings::eColor, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_DlssRRBindings.init(d, m_device, 1, 0, 0));
    NVVK_DBG_NAME(m_DlssRRBindings.getLayout());
  }

  void writeDlssSet()
  {
    nvvk::WriteSetContainer writes;

    auto appendWriteBindImage = [&](shaderio::DlssBindings binding, RenderBufferName gbuf) {
      writes.append(m_DlssRRBindings.makeWrite(binding), &m_renderBuffers.getDescriptorImageInfo(gbuf));
    };

    appendWriteBindImage(shaderio::DlssBindings::eBaseColor_Metalness, eGBufBaseColor_Metalness);
    appendWriteBindImage(shaderio::DlssBindings::eSpecAlbedo, eGBufSpecAlbedo);
    appendWriteBindImage(shaderio::DlssBindings::eSpecHitDist, eGBufSpecHitDist);
    appendWriteBindImage(shaderio::DlssBindings::eNormal_Roughness, eGBufNormalRoughness);
    appendWriteBindImage(shaderio::DlssBindings::eViewZ, eGBufViewZ);
    appendWriteBindImage(shaderio::DlssBindings::eMotionVectors, eGBufMotionVectors);
    appendWriteBindImage(shaderio::DlssBindings::eColor, eGBufColor);

    vkUpdateDescriptorSets(m_device, writes.size(), writes.data(), 0, nullptr);
  }


  void createRtxSet()
  {
    m_rtBindings.deinit();

    nvvk::DescriptorBindings d;

    // This descriptor set, holds the top level acceleration structure and the output image
    d.addBinding(shaderio::RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_rtBindings.init(d, m_device));
    NVVK_DBG_NAME(m_rtBindings.getLayout());
  }

  void writeRtxSet()
  {
    if(!m_scene.valid())
    {
      return;
    }

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx.tlas();

    nvvk::WriteSetContainer writes;
    writes.append(m_rtBindings.makeWrite(shaderio::RtxBindings::eTlas), tlas);

    vkUpdateDescriptorSets(m_device, writes.size(), writes.data(), 0, nullptr);
  }


  void createSceneSet()
  {
    m_sceneBindings.deinit();

    nvvk::DescriptorBindings d;

    // This descriptor set, holds the top level acceleration structure and the output image
    d.addBinding(shaderio::SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk.nbTextures(),
                 VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_sceneBindings.init(d, m_device));
    NVVK_DBG_NAME(m_sceneBindings.getLayout());
  }

  void writeSceneSet()
  {
    if(!m_scene.valid())
    {
      return;
    }

    nvvk::WriteSetContainer writes;

    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk.textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.append(m_sceneBindings.makeWrite(shaderio::SceneBindings::eTextures), diit.data());

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = 0; }

  void windowTitle()
  {
    // Window Title
    static float dirty_timer = 0.0F;
    dirty_timer += ImGui::GetIO().DeltaTime;
    if(dirty_timer > 1.0F)  // Refresh every seconds
    {
      const auto&           size = m_app->getViewportSize();
      std::array<char, 256> buf{};
      snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", TARGET_NAME,
               static_cast<int>(size.width), static_cast<int>(size.height), static_cast<int>(ImGui::GetIO().Framerate),
               1000.F / ImGui::GetIO().Framerate, m_frame);
      glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
      dirty_timer = 0;
    }
  }


  //--------------------------------------------------------------------------------------------------
  // Send a ray under mouse coordinates, and retrieve the information
  // - Set new camera interest point on hit position
  //
  void screenPicking()
  {
    auto* tlas = m_sceneRtx.tlas();
    if(tlas == VK_NULL_HANDLE)
      return;

    ImGui::Begin("Viewport");  // ImGui, picking within "viewport"
    auto  mouse_pos        = ImGui::GetMousePos();
    auto  main_size        = ImGui::GetContentRegionAvail();
    auto  corner           = ImGui::GetCursorScreenPos();  // Corner of the viewport
    float aspect_ratio     = main_size.x / main_size.y;
    mouse_pos              = mouse_pos - corner;
    ImVec2 local_mouse_pos = mouse_pos / main_size;
    ImGui::End();

    auto* cmd = m_app->createTempCmdBuffer();

    // Finding current camera matrices
    const auto& view = m_cameraManip->getViewMatrix();
    auto        proj = glm::perspectiveRH_ZO(glm::radians(m_cameraManip->getFov()), aspect_ratio, 0.1F, 1000.0F);
    proj[1][1] *= -1;

    // Setting up the data to do picking
    nvvk::RayPicker::PickInfo pick_info;
    pick_info.pickPos        = {local_mouse_pos.x, local_mouse_pos.y};
    pick_info.modelViewInv   = glm::inverse(view);
    pick_info.perspectiveInv = glm::inverse(proj);
    pick_info.tlas           = m_sceneRtx.tlas();

    // Run and wait for result
    m_picker.run(cmd, pick_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Retrieving picking information
    nvvk::RayPicker::PickResult pr = m_picker.getResult();
    if(pr.instanceID == ~0)
    {
      LOGI("Nothing Hit\n");
      return;
    }

    if(pr.hitT <= 0.F)
    {
      LOGI("Hit Distance == 0.0\n");
      return;
    }

    // Find where the hit point is and set the interest position
    glm::vec3 world_pos = glm::vec3(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    m_cameraManip->getLookat(eye, center, up);
    m_cameraManip->setLookat(eye, world_pos, up, false);

    // Logging picking info.
    const nvvkgltf::RenderNode& renderNode = m_scene.getRenderNodes()[pr.instanceID];
    const tinygltf::Node&       node       = m_scene.getModel().nodes[renderNode.refNodeID];

    LOGI("Node Name: %s\n", node.name.c_str());
    LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pr.primitiveID);
    LOGI(" - Render: GltfRenderNode: %d, RenderPrim: %d\n", pr.instanceID, pr.instanceCustomIndex);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
  }

  void raytraceScene(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);

    vkCmdUpdateBuffer(cmd, m_skyParamBuffer.buffer, 0, sizeof(m_skyParams), &m_skyParams);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtBindings.getSet(0), m_sceneBindings.getSet(0),
                                           m_DlssRRBindings.getSet(0), m_hdrEnv.getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);


    m_pushConst.frameInfo = (shaderio::FrameInfo*)m_bFrameInfo.address;
    m_pushConst.gltfScene = (shaderio::GltfScene*)m_sceneVk.sceneDesc().address;
    m_pushConst.skyParams = (shaderio::SkyPhysicalParameters*)m_skyParamBuffer.address;
    vkCmdPushConstants(cmd, m_rtPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtxPushConstant), &m_pushConst);

    const auto& size = m_renderBuffers.getSize();

    const auto& sbtRegions = m_sbt.getSBTRegions(0);
    vkCmdTraceRaysKHR(cmd, &sbtRegions.raygen, &sbtRegions.miss, &sbtRegions.hit, &sbtRegions.callable, size.width, size.height, 1);
  }

  void createHdr(const std::filesystem::path& filename)
  {
    auto cmd = m_app->createTempCmdBuffer();
    m_hdrEnv.destroyEnvironment();
    m_hdrEnv.loadEnvironment(cmd, m_stagingUploader, filename);
    m_stagingUploader.cmdUploadAppended(cmd);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_stagingUploader.releaseStaging();
  }

  void destroyResources()
  {
    m_dlss.deinit();
    m_ngx.deinit();

    m_alloc.destroyBuffer(m_bFrameInfo);

    m_sceneRtx.deinit();
    m_sceneVk.deinit();
    m_scene.destroy();

    m_hdrEnv.deinit();
    m_skyEnv.deinit();
    m_alloc.destroyBuffer(m_skyParamBuffer);

    m_renderBuffers.deinit();
    m_outputBuffers.deinit();

    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    m_rtPipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
    m_rtPipelineLayout = VK_NULL_HANDLE;

    m_rtBindings.deinit();
    m_sceneBindings.deinit();
    m_DlssRRBindings.deinit();

    m_alloc.destroyBuffer(m_sbtBuffer);
    m_sbt.deinit();

    m_picker.deinit();
    m_tonemapper.deinit();
    m_samplerPool.deinit();

    m_stagingUploader.deinit();
    m_alloc.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  VkDevice m_device = VK_NULL_HANDLE;

  nvapp::Application*     m_app{nullptr};
  nvvk::ResourceAllocator m_alloc{};  // The VMA allocator
  nvvk::StagingUploader   m_stagingUploader{};

  glm::uvec2 m_renderSize = {1, 1};
  glm::uvec2 m_outputSize = {1, 1};

  //#DLSS
  nvvk::GBuffer m_renderBuffers;  // lower render resolution
  nvvk::GBuffer m_outputBuffers;  // upscaled output resolution

  nvvk::DescriptorPack m_DlssRRBindings;  // DLSS render buffers descriptor set

  NgxContext                                     m_ngx;
  DlssRR                                         m_dlss;
  NVSDK_NGX_PerfQuality_Value                    m_dlssQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  NVSDK_NGX_RayReconstruction_Hint_Render_Preset m_dlssPreset  = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default;
  NgxContext::SupportedSizes                     m_dlssSizes;
  // UI options
  bool                                   m_dlssShowScaledBuffers = true;
  std::array<bool, DlssRR::RESOURCE_NUM> m_dlssBufferEnable;

  // Resources
  nvvk::Buffer m_bFrameInfo;

  // Pipeline
  shaderio::RtxPushConstant m_pushConst{
      -1,      // frame
      1000.f,  // maxLuminance for firefly checks
      7,       // max ray recursion
      1.0,     // meterToUnitsMultiplier
      -1.0,    // overrideRoughness
      -1.0,    // overrideMetallic
      {0, 0},  // mouseVec
      1.0,     // bitangentFlip
  };  // Information sent to the shader

  int m_frame{0};

  nvvk::DescriptorPack m_sceneBindings;  // Scene geometry, material and texture descriptors

  nvvk::DescriptorPack    m_rtBindings{};
  nvvk::WriteSetContainer m_rtWriteSetContainer{};

  VkPipelineLayout m_rtPipelineLayout{};  // The pipeline layout use with graphics pipeline
  VkPipeline       m_rtPipeline{};        // The pipeline

  //FIXME: there is no reason that we must pass m_cameraManip around as a shared_ptr excepto for the CameraWidget wills it so.
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip;

  shaderio::FrameInfo m_frameInfo{.flags = FLAGS_USE_PSR | FLAGS_USE_PATH_REGULARIZATION};

  nvvkgltf::Scene    m_scene;
  nvvkgltf::SceneVk  m_sceneVk;
  nvvkgltf::SceneRtx m_sceneRtx;

  nvvk::SBTGenerator m_sbt;  // Shading binding table wrapper
  nvvk::Buffer       m_sbtBuffer;

  nvvk::RayPicker   m_picker;  // For ray picking info
  nvvk::HdrIbl      m_hdrEnv;
  nvvk::SamplerPool m_samplerPool;  // HdrEnvDome wants this

  nvshaders::SkyPhysical          m_skyEnv;
  shaderio::SkyPhysicalParameters m_skyParams;
  nvvk::Buffer                    m_skyParamBuffer;

  nvshaders::Tonemapper    m_tonemapper;
  shaderio::TonemapperData m_tonemapperData;


  RenderBufferName m_showBuffer = eNumRenderBufferNames;
};

//////////////////////////////////////////////////////////////////////////
int main(int, char**)
{
  nvapp::ApplicationCreateInfo appInitInfo;
  appInitInfo.name  = TARGET_NAME " Example";
  appInitInfo.vSync = true;
  // spec.headless = true;
  // spec.headlessFrameCount = 10;

  if(appInitInfo.headless)
  {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
  }

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR    ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  nvvk::ContextInitInfo ctxInfo{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},

      .deviceExtensions = {{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME},
                           {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature},
                           {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature},
                           {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                           {VK_KHR_RAY_QUERY_EXTENSION_NAME, &ray_query_features, appInitInfo.headless == true},
                           {VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeature},
                           {VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME},
                           {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                           {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeature},
                           {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
  };

#if NVVK_SUPPORTS_AFTERMATH
  // Optional extension to support Aftermath shader level debugging
  ctxInfo.deviceExtension.emplace_back({VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME, true});
#endif

  nvvk::addSurfaceExtensions(ctxInfo.instanceExtensions);

  nvvk::ValidationSettings validation{};
  {
    // Enable Debug stuff
    validation.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);

    // Danger: keep validation alive until after vkCtx.init()
    ctxInfo.instanceCreateInfoExt = validation.buildPNextChain();

    g_dbgPrintf = std::make_shared<nvapp::ElementDbgPrintf>();
  }


  //#DLSS_RR determine required instance extensions
  std::vector<VkExtensionProperties> instanceExts;
  {
    NGX_ABORT_ON_FAIL(NgxContext::getDlssRRRequiredInstanceExtensions(instanceExts));
    for(const auto& e : instanceExts)
    {
      ctxInfo.instanceExtensions.emplace_back(e.extensionName);
    }
  }

  ctxInfo.preSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice) {
    return NVSDK_NGX_SUCCEED(NgxContext::isDlssRRAvailable(instance, physicalDevice));
  };
  ctxInfo.postSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& info) {
    static std::vector<VkExtensionProperties> dlssrrExtensions;
    NGX_CHECK(NgxContext::getDlssRRRequiredDeviceExtensions(instance, physicalDevice, dlssrrExtensions));
    for(const auto& e : dlssrrExtensions)
    {
      info.deviceExtensions.push_back({.extensionName = e.extensionName, .specVersion = e.specVersion});
    }

    return true;
  };

  // We need one queue. This queue will have "queue family index 0"
  ctxInfo.queues = {VK_QUEUE_GRAPHICS_BIT};

  nvvk::Context vkCtx;
  if(vkCtx.init(ctxInfo) != VK_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  appInitInfo.instance       = vkCtx.getInstance();
  appInitInfo.physicalDevice = vkCtx.getPhysicalDevice();
  appInitInfo.device         = vkCtx.getDevice();
  appInitInfo.queues.push_back(vkCtx.getQueueInfo(0));

  // Create the application
  nvapp::Application app;
  app.init(appInitInfo);

  // Create application elements
  std::shared_ptr<nvapp::IAppElement> dlss_applet = std::make_shared<DlssApplet>();
  g_elem_camera                                   = std::make_shared<nvapp::ElementCamera>();

  app.addElement(g_elem_camera);
  app.addElement(dlss_applet);
  app.addElement(g_dbgPrintf);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit

  // Search paths
  std::vector<std::filesystem::path> default_search_paths = {
      ".", "..", "../..", "../../..", nvutils::getExecutablePath().parent_path() / TARGET_EXE_TO_DOWNLOAD_DIRECTORY};

  // Load HDR
  std::filesystem::path hdr_file = nvutils::findFile(R"(environment.hdr)", default_search_paths);
  dlss_applet->onFileDrop(hdr_file);

  // Load scene
  std::filesystem::path scn_file = nvutils::findFile(R"(ABeautifulGame/glTF/ABeautifulGame.gltf)", default_search_paths);
  dlss_applet->onFileDrop(scn_file);

  // Run as fast as possible, without waiting for display vertical syncs.
  app.setVsync(false);

  app.run();
  app.deinit();
  dlss_applet.reset();
  g_elem_camera.reset();
  g_dbgPrintf.reset();

  vkCtx.deinit();

  return EXIT_SUCCESS;
}
