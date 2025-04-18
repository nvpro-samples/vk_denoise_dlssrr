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

#ifndef DLSS_RR_GLSL

#include "nvvkhl/shaders/func.h"

#define rsqrt inversesqrt
#define saturate(x) clamp(x, 0.0, 1.0)
#define lerp mix
#define mul(x, y) (x * y)

#define DLSS_INF_DISTANCE 65504.0  // FP16 max number

const float FLT_MIN = 1e-15;
float       PositiveRcp(float x)
{
  return 1.0 / (max(x, FLT_MIN));
}

// "Ray Tracing Gems", Chapter 32, Equation 4 - the approximation assumes GGX VNDF and Schlick's approximation
vec3 EnvironmentTerm_Rtg(vec3 Rf0, float NoV, float alphaRoughness)
{
  vec4 X;
  X.x = 1.0;
  X.y = NoV;
  X.z = NoV * NoV;
  X.w = NoV * X.z;

  vec4 Y;
  Y.x = 1.0;
  Y.y = alphaRoughness;
  Y.z = alphaRoughness * alphaRoughness;
  Y.w = alphaRoughness * Y.z;

  mat2 M1 = mat2(0.99044, -1.28514, 1.29678, -0.755907);
  mat3 M2 = mat3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627);

  mat2 M3 = mat2(0.0365463, 3.32707, 9.0632, -9.04756);
  mat3 M4 = mat3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123);

  float bias  = dot(mul(M1, X.xy), Y.xy) * PositiveRcp(dot(mul(M2, X.xyw), Y.xyw));
  float scale = dot(mul(M3, X.xy), Y.xy) * PositiveRcp(dot(mul(M4, X.xzw), Y.xyw));

  return saturate(Rf0 * scale + bias);
}

#endif
