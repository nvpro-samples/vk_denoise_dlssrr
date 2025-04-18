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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "dlss_helper.glsl"

#include "host_device.h"
#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/func.h"
#include "ray_common.glsl"

// clang-format off
layout(location = PAYLOAD_SECONDARY) rayPayloadInEXT PayloadSecondary payload;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 3, binding = eHdr) uniform sampler2D hdrTexture;
layout(set = 4, binding = eSkyParam)  uniform SkyInfo_ { PhysicalSkyParameters skyInfo; };

// clang-format on

// If the pathtracer misses, it means the ray segment hit the environment map.
void main()
{
  vec3  envColor;
  float envPdf;

  if(TEST_FLAG(frameInfo.flags, FLAGS_ENVMAP_SKY))
  {
    envColor = evalPhysicalSky(skyInfo, gl_WorldRayDirectionEXT);
    envPdf   = samplePhysicalSkyPDF(skyInfo, gl_WorldRayDirectionEXT);
  }
  else
  {
    vec3 dir         = rotate(gl_WorldRayDirectionEXT, vec3(0, 1, 0), -frameInfo.envRotation);
    vec2 uv          = getSphericalUv(dir);
    vec4 hdrColorPdf = texture(hdrTexture, uv);
    envColor         = hdrColorPdf.rgb;
    envPdf           = hdrColorPdf.w;
  }

  envColor *= frameInfo.envIntensity.xyz;

  // From any surface point its possible to hit the environment map via two ways
  // a) as result from direct sampling or b) as result of following the material
  // BSDF. Here we deal with b). Calculate the proper MIS weight by taking the
  // BSDF's PDF in ray direction and the envmap's PDF in ray direction into account.
  float mis_weight = powerHeuristic(payload.bsdfPDF, envPdf);
  payload.contrib  = mis_weight * envColor;
  payload.hitT     = -DLSS_INF_DISTANCE;  // Ending trace
}
