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

#include "host_device.h"
#include "dlss_helper.glsl"
#include "nvvkhl/shaders/func.h"
#include "ray_common.glsl"

#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_sky.h"

// clang-format off
layout(location = 0) rayPayloadInEXT PayloadPrimary payloadPrimary;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 3, binding = eHdr) uniform sampler2D hdrTexture;
layout(set = 4, binding = eSkyParam)  uniform SkyInfo_ { PhysicalSkyParameters skyInfo; };

// clang-format on

// The main miss shader will be executed when the primaary rays misses any geometry
// and just hit the background envmap. The resulting color values will be recorded
// into the "DirectLighting" buffer.
void main()
{
  vec3 envColor;

  if(TEST_FLAG(frameInfo.flags, FLAGS_ENVMAP_SKY))
  {
    envColor = evalPhysicalSky(skyInfo, gl_WorldRayDirectionEXT);
  }
  else
  {
    vec3 camOffset = 1.0 + vec3(frameInfo.view[3]);

    vec3 dir = gl_WorldRayDirectionEXT;

    // dir.xz *= 1.0 / camOffset.y;
    // dir.y *= camOffset.y;
    // dir = normalize(dir);

    dir = rotate(dir, vec3(0, 1, 0), -frameInfo.envRotation);


    vec2 uv  = getSphericalUv(dir);
    envColor = texture(hdrTexture, uv).rgb;
  }

  // No need to deal with the PDF here since the primary surface trace is
  // performed noise-free.
  payloadPrimary.normal_envmapRadiance = envColor * frameInfo.envIntensity.xyz;
  payloadPrimary.hitT                  = DLSS_INF_DISTANCE;  // Ending trace
}
