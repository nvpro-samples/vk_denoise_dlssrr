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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#define GRID_SIZE 16  // Grid size used by compute shaders

// clang-format off
#ifdef __cplusplus
  #include <glm/glm.hpp>
  using uint = uint32_t;

  using mat4 = glm::mat4;
  using vec4 = glm::vec4;
  using vec3 = glm::vec3;
  using vec2 = glm::vec2;
  using ivec2 = glm::ivec2;
  using uint = uint32_t;
#endif  // __cplusplus

#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

#define NB_LIGHTS 0

// We have two sets of shaders compiled into the Shader Binding Table;
// primary shaders, light-weight shaders used when finding the primary surface
// (which doesn't require random sampling), and pathtrace shaders, which are
// used for Monte Carlo path tracing.
#define PAYLOAD_PRIMARY         0
#define PAYLOAD_SECONDARY   1
#define SBTOFFSET_PRIMARY       0
#define SBTOFFSET_SECONDARY 1
#define MISSINDEX_PRIMARY       0
#define MISSINDEX_SECONDARY 1

START_BINDING(SceneBindings)
  eFrameInfo = 0,
  eSceneDesc = 1,
  eTextures  = 2
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0
END_BINDING();

START_BINDING(PostBindings)
  ePostImage       = 0
END_BINDING();

START_BINDING(RTBindings)
  eViewZ            = 0,
  eMotionVectors     = 1,
  eNormal_Roughness = 2,
  eBaseColor_Metalness = 3,
  eSpecAlbedo       = 4,
  eColor            = 5,
  eSpecHitDist      = 6
END_BINDING();

START_BINDING(TaaBindings)
  eInImage = 0,
  eOutImage  = 1
END_BINDING();
// clang-format on

struct Light
{
  vec3  position;
  float intensity;
  vec3  color;
  int   type;
};

#define TEST_FLAG(flags, flag) bool((flags) & (flag))

#define BIT(x) (1 << x)

#define FLAGS_ENVMAP_SKY BIT(0)
#define FLAGS_USE_PSR BIT(1)
#define FLAGS_USE_PATH_REGULARIZATION BIT(2)


struct FrameInfo
{
  mat4  view;
  mat4  proj;
  mat4  viewInv;
  mat4  projInv;
  mat4  prevMVP;
  vec4  envIntensity;
  vec2  jitter;
  float envRotation;
  uint  flags;  // beware std430 layout requirements
#if NB_LIGHTS > 0
  Light light[NB_LIGHTS];
#endif
};

struct RtxPushConstant
{
  int   frame;
  float maxLuminance;
  uint  maxDepth;
  float meterToUnitsMultiplier;
  float overrideRoughness;
  float overrideMetallic;
  ivec2 mouseCoord;
  float bitangentFlip;
};

#ifdef __cplusplus
#include <vulkan/vulkan_core.h>

inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}
#endif

#endif  // HOST_DEVICE_H
