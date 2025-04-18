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

#include <array>
#include <filesystem>
#include <math.h>
#include <memory>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "backends/imgui_impl_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"

#include "nvh/fileoperations.hpp"
#include "nvh/gltfscene.hpp"
#include "nvp/nvpsystem.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_dbgprintf.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "nvvkhl/sky.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"


#include "_autogen/primary.rchit.h"
#include "_autogen/primary.rgen.h"
#include "_autogen/primary.rmiss.h"
#include "_autogen/secondary.rahit.h"
#include "_autogen/secondary.rchit.h"
#include "_autogen/secondary.rmiss.h"
#include "shaders/host_device.h"

#include <glm/gtc/type_ptr.hpp>

#include "dlssrr_wrapper.hpp"

std::shared_ptr<nvvkhl::ElementCamera>    g_elem_camera;
std::shared_ptr<nvvkhl::ElementDbgPrintf> g_dbgPrintf;

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


using namespace nvh::gltf;

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
class DlssApplet : public nvvkhl::IAppElement
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
  ~DlssApplet() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice         = app->getPhysicalDevice();
    allocator_info.device                 = app->getDevice();
    allocator_info.instance               = app->getInstance();
    allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);         // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(allocator_info);  // Allocator
    m_scene = std::make_unique<nvh::gltf::Scene>();                // GLTF scene
    m_sceneVk = std::make_unique<nvvkhl::SceneVk>(m_device, m_app->getPhysicalDevice(), m_alloc.get());  // GLTF Scene buffers
    m_sceneRtx = std::make_unique<nvvkhl::SceneRtx>(m_device, m_app->getPhysicalDevice(), m_alloc.get());  // GLTF Scene BLAS/TLAS
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_device, m_alloc.get());
    m_sbt        = std::make_unique<nvvk::SBTWrapper>();
    m_picker     = std::make_unique<nvvk::RayPickerKHR>(m_device, m_app->getPhysicalDevice(), m_alloc.get());
    m_hdrEnv     = std::make_unique<nvvkhl::HdrEnv>(m_device, m_app->getPhysicalDevice(), m_alloc.get());
    m_rtxSet     = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_sceneSet   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_DlssRRSet  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    m_skyEnv = std::make_unique<nvvkhl::PhysicalSkyDome>();
    m_skyEnv->setup(m_device, m_alloc.get());

    m_hdrEnv->loadEnvironment("");

    // Requesting ray tracing properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rt_prop;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);
    // Create utilities to create the Shading Binding Table (SBT)
    uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt->setup(m_app->getDevice(), gct_queue_index, m_alloc.get(), rt_prop);

    m_outputSize = {app->getWindowSize().width, app->getWindowSize().height};

    createVulkanBuffers();

    // #DLSS
    {
      NGX_ABORT_ON_FAIL(m_ngx.init({.instance        = m_app->getInstance(),
                                    .physicalDevice  = m_app->getPhysicalDevice(),
                                    .device          = m_app->getDevice(),
                                    .applicationPath = NVPSystem::exePath()}));

      if(NVSDK_NGX_FAILED(m_ngx.isDlssRRAvailable()))
      {
        LOGE("DLSS is not available, aborting.\n");
        exit(EXIT_FAILURE);
        return;
      }

      m_dlssBufferEnable.fill(true);

      reinitDlss(true);
    }

    // Create resources in DLSS_RR input render size and output size
    createInputGbuffers(m_renderSize);
    createInputGbuffers(m_outputSize);

    m_tonemapper->createComputePipeline();
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

    NGX_ABORT_ON_FAIL(m_ngx.initDlssRR({.inputSize  = {m_renderSize.x, m_renderSize.y},
                                        .outputSize = {m_outputSize.x, m_outputSize.y},
                                        .quality    = m_dlssQuality,
                                        .preset     = m_dlssPreset},
                                       m_dlss));

    createInputGbuffers(m_renderSize);

    writeRtxSet();
  }

  void setDlssResources()
  {
    auto dlssRenderResourceFromGBufTexture = [&](DlssRR::DlssResource dlssResource, RenderBufferName gbufIndex) {
      m_dlssBufferEnable[gbufIndex] ? m_dlss.setResource(dlssResource, m_renderBuffers->getColorImage(gbufIndex),
                                                         m_renderBuffers->getDescriptorImageInfo(gbufIndex).imageView,
                                                         m_renderBuffers->getColorFormat(gbufIndex)) :
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
      m_dlss.setResource(dlssResource, m_outputBuffers->getColorImage(gbufIndex),
                         m_outputBuffers->getDescriptorImageInfo(gbufIndex).imageView, m_outputBuffers->getColorFormat(gbufIndex));
    };
    dlssOutputResourceFromGBufTexture(DlssRR::RESOURCE_COLOR_OUT, eGBufColorOut);
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    vkDeviceWaitIdle(m_device);

    m_outputSize = {width, height};
    // #DLSS
    // Work around a bug in DLSS_RR that causes a crash below a certain image size
    m_outputSize = glm::max({256, 256}, m_outputSize);

    createOutputGbuffer(m_outputSize);
    writeRtxSet();

    reinitDlss(true);

    m_tonemapper->updateComputeDescriptorSets(m_outputBuffers->getDescriptorImageInfo(eGBufColorOut),
                                              m_outputBuffers->getDescriptorImageInfo(eGBufLdr));
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
      auto filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                      "glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
      onFileDrop(filename.c_str());
    }
  }

  void onFileDrop(const char* filename) override
  {
    namespace fs = std::filesystem;
    vkDeviceWaitIdle(m_device);
    std::string extension = fs::path(filename).extension().string();
    if(extension == ".gltf" || extension == ".glb")
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
    using namespace ImGuiH;

    bool reset{false};
    // Pick under mouse cursor
    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
    {
      screenPicking();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_M))
    {
      onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);  // Force recreation of G-Buffers
      reset = true;
    }

    {  // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::CollapsingHeader("Camera"))
      {
        ImGuiH::CameraWidget();
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
        m_pushConst.bitangentFlip = flipBitangent ? -1.0 : 1.0;

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
          m_skyEnv->onUI();
        }

        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        m_tonemapper->onUI();
      }

      if(ImGui::CollapsingHeader("DLSS RR", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        {
          {  // Note that UltraQuality is deliberately left out as unsupported, see DLSS_RR Integration Guide
            const char* const items[] = {"MaxPerf", "Balanced", "MaxQuality", "UltraPerformance", "DLAA"};
            NVSDK_NGX_PerfQuality_Value itemValues[]{NVSDK_NGX_PerfQuality_Value_MaxPerf, NVSDK_NGX_PerfQuality_Value_Balanced,
                                                     NVSDK_NGX_PerfQuality_Value_MaxQuality, NVSDK_NGX_PerfQuality_Value_UltraPerformance,
                                                     NVSDK_NGX_PerfQuality_Value_DLAA};
            // Find item corresponding to currently selected quality
            int item;
            for(item = 0; item < arraySize(items) && itemValues[item] != m_dlssQuality; ++item)
              ;
            if(PropertyEditor::entry("Quality", [&]() {
                 return ImGui::ListBox("Quality", &item, items, arraySize(items), 3 /*heightInItems*/);
               }))
            {
              m_dlssQuality = itemValues[item];
              reinitDlss(true);
              reset = true;
            }
          }

          {  // Some of the presets are marked as "Do not use". See nvsdk_ngx_defs.h
            const char* const                 items[]      = {"Default",  "Preset A", "Preset B", "Preset C",
                                                              "Preset D", "Preset E", "Preset F", "Preset J"};
            NVSDK_NGX_DLSS_Hint_Render_Preset itemValues[] = {
                NVSDK_NGX_DLSS_Hint_Render_Preset_Default,  // default behavior, may or may not change after OTA
                NVSDK_NGX_DLSS_Hint_Render_Preset_A,       NVSDK_NGX_DLSS_Hint_Render_Preset_B,
                NVSDK_NGX_DLSS_Hint_Render_Preset_C,       NVSDK_NGX_DLSS_Hint_Render_Preset_D,
                NVSDK_NGX_DLSS_Hint_Render_Preset_E,       NVSDK_NGX_DLSS_Hint_Render_Preset_F,
                NVSDK_NGX_DLSS_Hint_Render_Preset_J};

            // Find item corresponding to currently selected preset
            int item;
            for(item = 0; item < arraySize(items) && itemValues[item] != m_dlssPreset; ++item)
              ;
            if(PropertyEditor::entry("Presets", [&]() {
                 return ImGui::ListBox("Presets", &item, items, arraySize(items), 3 /*heightInItems*/);
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

        ImVec2 tumbnailSize = {100 * m_renderBuffers->getAspectRatio(), 100};

        auto showBuffer = [&](const char* name, RenderBufferName buffer, bool optional = false) {
          ImGui::PushID(name);
          ImGui::TableNextColumn();
          if(ImGui::ImageButton(name, m_renderBuffers->getDescriptorSet(buffer), tumbnailSize))
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
          if(ImGui::ImageButton("Denoised", m_outputBuffers->getDescriptorSet(eGBufLdr), tumbnailSize))
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

      ImVec2 imageSize = m_dlssShowScaledBuffers ? ImGui::GetContentRegionAvail() : ImVec2(m_renderSize.x, m_renderSize.y);
      // Display the G-Buffer image in the main viewport
      (m_showBuffer == eNumRenderBufferNames) ?
          ImGui::Image(m_outputBuffers->getDescriptorSet(eGBufLdr), ImGui::GetContentRegionAvail()) :
          ImGui::Image(m_renderBuffers->getDescriptorSet(m_showBuffer), imageSize);

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Get camera info
    float view_aspect_ratio = (float)m_outputSize.x / m_outputSize.y;

    m_frameInfo.prevMVP = m_frameInfo.proj * m_frameInfo.view;

    // Update Frame buffer uniform buffer
    const auto& clip = CameraManip.getClipPlanes();
    m_frameInfo.view = CameraManip.getMatrix();
    m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);

    // Were're feeding the raytracer with a flipped matrix for convenience
    m_frameInfo.proj[1][1] *= -1;

    m_frameInfo.projInv      = glm::inverse(m_frameInfo.proj);
    m_frameInfo.viewInv      = glm::inverse(m_frameInfo.view);
    m_frameInfo.envRotation  = m_settings.envRotation;
    m_frameInfo.envIntensity = m_settings.envIntensity;
    m_frameInfo.jitter       = halton(m_frame) - vec2(0.5);

    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

    // Push constant
    m_pushConst.maxDepth   = m_settings.maxDepth;
    m_pushConst.frame      = m_frame;
    m_pushConst.mouseCoord = g_dbgPrintf->getMouseCoord();

    auto renderBufferShaderWriteToRead = [this](RenderBufferName buffer) {
      return nvvk::makeImageMemoryBarrier(m_renderBuffers->getColorImage(buffer), VK_ACCESS_SHADER_WRITE_BIT,
                                          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    };
    auto outputBufferShaderWriteToRead = [this](OutputBufferName buffer) {
      return nvvk::makeImageMemoryBarrier(m_outputBuffers->getColorImage(buffer), VK_ACCESS_SHADER_WRITE_BIT,
                                          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    };

    auto renderBufferShaderReadToWrite = [this](RenderBufferName buffer) {
      return nvvk::makeImageMemoryBarrier(m_renderBuffers->getColorImage(buffer), VK_ACCESS_SHADER_READ_BIT,
                                          VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    };

    auto outputBufferShaderReadToShaderWrite = [this](OutputBufferName buffer) {
      return nvvk::makeImageMemoryBarrier(m_outputBuffers->getColorImage(buffer), VK_ACCESS_SHADER_READ_BIT,
                                          VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    };


    {
      std::vector<VkImageMemoryBarrier> barriers{renderBufferShaderReadToWrite(eGBufColor)};

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, barriers.size(), barriers.data());
    }

    // Pathtrace the scene
    raytraceScene(cmd);

    {
      std::vector<VkImageMemoryBarrier> barriers{renderBufferShaderWriteToRead(eGBufColor),
                                                 outputBufferShaderReadToShaderWrite(eGBufColorOut)};

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, barriers.size(), barriers.data());
    }

    // #DLSS
    setDlssResources();
    // Check, but don't exit here, because we can disable non-optional guide buffers
    NGX_CHECK(m_dlss.denoise(cmd, m_renderSize, m_frameInfo.jitter, m_frameInfo.view, m_frameInfo.proj, m_frame == 0));

    {
      std::vector<VkImageMemoryBarrier> barriers{outputBufferShaderWriteToRead(eGBufColorOut),
                                                 outputBufferShaderReadToShaderWrite(eGBufLdr)};

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, barriers.size(), barriers.data());
    }


    // Apply tonemapper - take GBuffer-X and output to GBuffer-0
    m_tonemapper->runCompute(cmd, m_outputBuffers->getSize());

    m_frame++;
  }


  void onLastHeadlessFrame() override
  {
    // FIXME: output size is probably not the right one
    nvvkhl::GBuffer temp(m_app->getDevice(), m_alloc.get(), m_app->getWindowSize(), VK_FORMAT_R8G8B8A8_UNORM);
    // Image to render to
    const VkRenderingAttachmentInfoKHR colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView   = temp.getColorImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,   // Clear the image (see clearValue)
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,  // Store the image (keep the image)
        .clearValue  = {{{1.0f, 0.0f, 0.0f, 1.0f}}},
    };

    // Details of the dynamic rendering
    const VkRenderingInfoKHR renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .renderArea           = {{0, 0}, temp.getSize()},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
    };

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Transition the swapchain image to the color attachment layout, needed when using dynamic rendering
    nvvk::cmdBarrierImageLayout(cmd, temp.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);


    vkCmdBeginRendering(cmd, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, temp.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    vkDeviceWaitIdle(m_device);

    m_app->saveImageToFile(temp.getColorImage(), temp.getSize(),
                           nvh::getExecutablePath().replace_extension(".screenshot.jpg").string(), 95);
  };

private:
  void createScene(const std::string& filename)
  {
    if(!m_scene->load(filename))
    {
      LOGE("Error loading scene");
      return;
    }

    nvvkhl::setCamera(filename, m_scene->getRenderCameras(), m_scene->getSceneBounds());  // Camera auto-scene-fitting
    g_elem_camera->setSceneRadius(m_scene->getSceneBounds().radius());                    // Navigation help

    {  // Create the Vulkan side of the scene
      auto cmd = m_app->createTempCmdBuffer();
      m_sceneVk->create(cmd, *m_scene);
      m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);  // Create BLAS / TLAS

      m_app->submitAndWaitTempCmdBuffer(cmd);

      m_picker->setTlas(m_sceneRtx->tlas());
    }

    // Descriptor Set and Pipelines
    createSceneSet();
    createRtxSet();
    createDLSSSet();
    createRtxPipeline();  // must recreate due to texture changes
    writeSceneSet();
    writeRtxSet();
  }

  void createInputGbuffers(const glm::uvec2& inputSize)
  {
    // Creation of the GBuffers
    m_renderBuffers.reset();

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

    m_renderBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), vk_size, colorBuffers, VK_FORMAT_UNDEFINED);

    // Indicate the renderer to reset its frame
    resetFrame();
  }

  void createOutputGbuffer(const glm::uvec2& outputSize)
  {
    m_outputBuffers.reset();

    VkExtent2D vk_size{outputSize.x, outputSize.y};

    std::vector<VkFormat> colorBuffers(eNumOutputBufferNames);
    colorBuffers[eGBufLdr] = VK_FORMAT_R8G8B8A8_UNORM;

    // #DLSS
    colorBuffers[eGBufColorOut] = VK_FORMAT_R16G16B16A16_SFLOAT;

    // Creation of the GBuffers
    m_outputBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), vk_size, colorBuffers, VK_FORMAT_UNDEFINED);

    // Indicate the renderer to reset its frame
    resetFrame();
  }

  // Create all Vulkan buffer data
  void createVulkanBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createRtxSet()
  {
    auto& d = m_rtxSet;
    d->deinit();
    d->init(m_device);

    // This descriptor set, holds the top level acceleration structure and the output image
    d->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);

    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  void createSceneSet()
  {
    auto& d = m_sceneSet;
    d->deinit();
    d->init(m_device);

    // This descriptor set, holds the top level acceleration structure and the output image
    d->addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk->nbTextures(), VK_SHADER_STAGE_ALL);
    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  void createDLSSSet()
  {
    auto& d = m_DlssRRSet;
    d->deinit();
    d->init(m_device);

    // #DLSS_RR
    d->addBinding(RTBindings::eNormal_Roughness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RTBindings::eBaseColor_Metalness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RTBindings::eSpecAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RTBindings::eSpecHitDist, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RTBindings::eViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RTBindings::eMotionVectors, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RTBindings::eColor, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);

    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    auto& p = m_rtxPipe;
    p.destroy(m_device);
    p.plines.resize(1);
    // Creating all shaders
    enum StageIndices
    {
      ePrimaryRaygen,
      ePrimaryHit,
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
    stage.module           = nvvk::createShaderModule(m_device, primary_rgen, sizeof(primary_rgen));
    stage.stage            = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[ePrimaryRaygen] = stage;
    // Miss
    stage.module           = nvvk::createShaderModule(m_device, secondary_rmiss, sizeof(secondary_rmiss));
    stage.stage            = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eSecondaryMiss] = stage;
    stage.module           = nvvk::createShaderModule(m_device, primary_rmiss, sizeof(primary_rmiss));
    stage.stage            = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[ePrimaryMiss]   = stage;
    // Hit Group - Closest Hit
    stage.module                 = nvvk::createShaderModule(m_device, secondary_rchit, sizeof(secondary_rchit));
    stage.stage                  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eSecondaryClosestHit] = stage;
    // AnyHit
    stage.module             = nvvk::createShaderModule(m_device, secondary_rahit, sizeof(secondary_rahit));
    stage.stage              = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eSecondaryAnyHit] = stage;
    stage.module             = nvvk::createShaderModule(m_device, primary_rchit, sizeof(primary_rchit));
    stage.stage              = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[ePrimaryHit]      = stage;
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
    group.closestHitShader = ePrimaryHit;
    group.anyHitShader     = eSecondaryAnyHit;
    shaderGroups.push_back(group);

    // Secondary closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eSecondaryClosestHit;
    group.anyHitShader     = eSecondaryAnyHit;
    shaderGroups.push_back(group);


    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtxSet->getLayout(), m_sceneSet->getLayout(),
                                                              m_DlssRRSet->getLayout(), m_hdrEnv->getDescriptorSetLayout(),
                                                              m_skyEnv->getDescriptorSetLayout()};
    pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts                = rt_desc_set_layouts.data();
    vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    ray_pipeline_info.pGroups                      = shaderGroups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data());
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt->create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
    for(auto& s : stages)
    {
      vkDestroyShaderModule(m_device, s.module, nullptr);
    }
  }

  void writeRtxSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx->tlas();
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;


    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eTlas, &desc_as_info));

    // #DLSS images that the RTX pipeline produces
    auto bindImage = [&](RTBindings binding, RenderBufferName gbuf) {
      writes.emplace_back(m_DlssRRSet->makeWrite(0, binding, &m_renderBuffers->getDescriptorImageInfo(gbuf)));
    };

    bindImage(RTBindings::eBaseColor_Metalness, eGBufBaseColor_Metalness);
    bindImage(RTBindings::eSpecAlbedo, eGBufSpecAlbedo);
    bindImage(RTBindings::eSpecHitDist, eGBufSpecHitDist);
    bindImage(RTBindings::eNormal_Roughness, eGBufNormalRoughness);
    bindImage(RTBindings::eViewZ, eGBufViewZ);
    bindImage(RTBindings::eMotionVectors, eGBufMotionVectors);
    bindImage(RTBindings::eColor, eGBufColor);

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }


  void writeSceneSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto& d = m_sceneSet;

    // Write to descriptors
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo scene_desc{m_sceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, SceneBindings::eFrameInfo, &dbi_unif));
    writes.emplace_back(d->makeWrite(0, SceneBindings::eSceneDesc, &scene_desc));
    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk->textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(d->makeWriteArray(0, SceneBindings::eTextures, diit.data()));

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
      snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", PROJECT_NAME,
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
    auto* tlas = m_sceneRtx->tlas();
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
    const auto& view = CameraManip.getMatrix();
    auto        proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, 0.1F, 1000.0F);
    proj[1][1] *= -1;

    // Setting up the data to do picking
    nvvk::RayPickerKHR::PickInfo pick_info;
    pick_info.pickX          = local_mouse_pos.x;
    pick_info.pickY          = local_mouse_pos.y;
    pick_info.modelViewInv   = glm::inverse(view);
    pick_info.perspectiveInv = glm::inverse(proj);

    // Run and wait for result
    m_picker->run(cmd, pick_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Retrieving picking information
    nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
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
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, world_pos, up, false);

    //    auto float_as_uint = [](float f) { return *reinterpret_cast<uint32_t*>(&f); };

    // Logging picking info.
    const nvh::gltf::RenderNode& renderNode = m_scene->getRenderNodes()[pr.instanceID];
    const tinygltf::Node&        node       = m_scene->getModel().nodes[renderNode.refNodeID];

    LOGI("Node Name: %s\n", node.name.c_str());
    LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pr.primitiveID);
    LOGI(" - Render: RenderNode: %d, RenderPrim: %d\n", pr.instanceID, pr.instanceCustomIndex);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
  }

  void raytraceScene(VkCommandBuffer cmd)
  {
    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    m_skyEnv->updateParameterBuffer(cmd);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtxSet->getSet(), m_sceneSet->getSet(), m_DlssRRSet->getSet(),
                                           m_hdrEnv->getDescriptorSet(), m_skyEnv->getDescriptorSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtxPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant), &m_pushConst);

    const auto& size = m_renderBuffers->getSize();

    auto sbtRegions = m_sbt->getRegions(0);
    vkCmdTraceRaysKHR(cmd, &sbtRegions[0], &sbtRegions[1], &sbtRegions[2], &sbtRegions[3], size.width, size.height, 1);
  }

  void createHdr(const char* filename)
  {
    m_hdrEnv = std::make_unique<nvvkhl::HdrEnv>(m_app->getDevice(), m_app->getPhysicalDevice(), m_alloc.get());

    m_hdrEnv->loadEnvironment(filename);
  }

  void destroyResources()
  {
    m_dlss.deinit();
    m_ngx.deinit();

    m_alloc->destroy(m_bFrameInfo);

    m_hdrEnv.reset();
    m_skyEnv.reset();

    m_renderBuffers.reset();
    m_outputBuffers.reset();

    m_rtxPipe.destroy(m_device);
    m_rtxSet->deinit();
    m_sceneSet->deinit();
    m_DlssRRSet->deinit();
    m_sbt->destroy();
    m_picker->destroy();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  VkDevice m_device = VK_NULL_HANDLE;

  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::unique_ptr<nvvkhl::AllocVma> m_alloc;

  glm::uvec2 m_renderSize = {1, 1};
  glm::uvec2 m_outputSize = {1, 1};

  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtxSet;    // TLAS descriptors
  std::unique_ptr<nvvk::DescriptorSetContainer> m_sceneSet;  // Scene geometry, material and texture descriptors

  //#DLSS
  std::unique_ptr<nvvkhl::GBuffer> m_renderBuffers;  // lower render resolution
  std::unique_ptr<nvvkhl::GBuffer> m_outputBuffers;  // upscaled ouput resolution

  std::unique_ptr<nvvk::DescriptorSetContainer> m_DlssRRSet;  // DLSS render buffers descriptor set
  NgxContext                                    m_ngx;
  DlssRR                                        m_dlss;
  NVSDK_NGX_PerfQuality_Value                   m_dlssQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  NVSDK_NGX_DLSS_Hint_Render_Preset             m_dlssPreset  = NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
  NgxContext::SupportedSizes                    m_dlssSizes;
  // UI options
  bool                                   m_dlssShowScaledBuffers = true;
  std::array<bool, DlssRR::RESOURCE_NUM> m_dlssBufferEnable;

  // Resources
  nvvk::Buffer m_bFrameInfo;

  // Pipeline
  RtxPushConstant m_pushConst{
      -1,      // frame
      1000.f,  // maxLuminance for firefly checks
      7,       // max ray recursion
      1.0,     // meterToUnitsMultiplier
      -1.0,    // overrideRoughness
      -1.0,    // overrideMetallic
      {0, 0},  // mouseVec
      1.0,     // bitangentFlip
  };  // Information sent to the shader

  int                       m_frame{0};
  nvvkhl::PipelineContainer m_rtxPipe;
  FrameInfo                 m_frameInfo{.flags = FLAGS_USE_PSR | FLAGS_USE_PATH_REGULARIZATION};

  std::unique_ptr<nvh::gltf::Scene>              m_scene;
  std::unique_ptr<nvvkhl::SceneVk>               m_sceneVk;
  std::unique_ptr<nvvkhl::SceneRtx>              m_sceneRtx;
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;
  std::unique_ptr<nvvk::SBTWrapper>              m_sbt;     // Shading binding table wrapper
  std::unique_ptr<nvvk::RayPickerKHR>            m_picker;  // For ray picking info
  std::unique_ptr<nvvkhl::HdrEnv>                m_hdrEnv;
  std::unique_ptr<nvvkhl::PhysicalSkyDome>       m_skyEnv;

  RenderBufferName m_showBuffer = eNumRenderBufferNames;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
int main(int, char**)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name  = PROJECT_NAME " Example";
  spec.vSync = true;
  // spec.headless = true;
  // spec.headlessFrameCount = 10;

  if(spec.headless)
  {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
  }

  nvvk::ContextCreateInfo ctxInfo;
  ctxInfo.apiMajor = 1;
  ctxInfo.apiMinor = 3;

  ctxInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  ctxInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, spec.headless, &ray_query_features);  // Used for picking

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeature);
  ctxInfo.addDeviceExtension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);

  // Display extension
  ctxInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
  ctxInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  ctxInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

#if NVVK_SUPPORTS_AFTERMATH
  // Optional extension to support Aftermath shader level debugging
  ctxInfo.addDeviceExtension(VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME, true);
#endif

  nvvkhl::addSurfaceExtensions(ctxInfo.instanceExtensions);

  g_dbgPrintf                   = std::make_shared<nvvkhl::ElementDbgPrintf>();
  ctxInfo.instanceCreateInfoExt = g_dbgPrintf->getFeatures();

  //#DLSS_RR determine required instance extensions
  std::vector<VkExtensionProperties> instanceExts;
  {
    NGX_ABORT_ON_FAIL(NgxContext::getDlssRRRequiredInstanceExtensions(instanceExts));
    for(auto e : instanceExts)
    {
      ctxInfo.addInstanceExtension(e.extensionName);
    }
  }

  nvvk::Context vkCtx;

  // Individual object creation
  if(!vkCtx.initInstance(ctxInfo))
  {
    LOGE("Vulkan Instance Creation failed.");
    return EXIT_FAILURE;
  }

  //#DLSS_RR determine required device extensions
  std::vector<VkExtensionProperties> deviceExts;
  {
    std::vector<VkPhysicalDevice> physDevices = vkCtx.getPhysicalDevices();
    NGX_ABORT_ON_FAIL(NgxContext::getDlssRRRequiredDeviceExtensions(vkCtx.m_instance, physDevices[0], deviceExts));
    for(auto e : deviceExts)
    {
      ctxInfo.addDeviceExtension(e.extensionName);
    }

    ctxInfo.removeDeviceExtension(VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  }

  // Find all compatible devices
  auto compatibleDevices = vkCtx.getCompatibleDevices(ctxInfo);
  if(compatibleDevices.empty())
  {
    LOGE("No compatible device found");
    return EXIT_FAILURE;
  }

  if(!vkCtx.initDevice(compatibleDevices[0], ctxInfo))
  {
    LOGE("ERROR: Vulkan Context Creation failed.");
    return EXIT_FAILURE;
  }

  // #DLSS_RR has an annoying bug that causes spamming Vulkan Validation Errors
#ifndef NDEBUG
  vkCtx.ignoreDebugMessage(0xeb0b9b05);
#endif

  spec.instance       = vkCtx.m_instance;
  spec.physicalDevice = vkCtx.m_physicalDevice;
  spec.device         = vkCtx.m_device;

  spec.queues.push_back({vkCtx.m_queueGCT.familyIndex, vkCtx.m_queueGCT.queueIndex, vkCtx.m_queueGCT.queue});

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create application elements
  auto dlss_applet = std::make_shared<DlssApplet>();
  g_elem_camera    = std::make_shared<nvvkhl::ElementCamera>();

  app->addElement(g_elem_camera);
  app->addElement(dlss_applet);
  app->addElement(g_dbgPrintf);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit

  // Search paths
  std::vector<std::string> default_search_paths = {".", "..", "../..", "../../..", NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY};

  // Load scene
  std::string scn_file = nvh::findFile(R"(ABeautifulGame/glTF/ABeautifulGame.gltf)", default_search_paths, true);
  dlss_applet->onFileDrop(scn_file.c_str());

  // Load HDR
  std::string hdr_file = nvh::findFile(R"(environment.hdr)", default_search_paths, true);
  dlss_applet->onFileDrop(hdr_file.c_str());

  // Run as fast as possible, without waiting for display vertical syncs.
  app->setVsync(false);

  app->run();
  app.reset();
  dlss_applet.reset();
  g_elem_camera.reset();
  g_dbgPrintf.reset();

  vkCtx.deinit();

  return EXIT_SUCCESS;
}
