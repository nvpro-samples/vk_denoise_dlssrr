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

#include "host_device.h"
#include "ray_common.slang"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/pbr_material_types.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/gltf_vertex_access.h.slang"
#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pushConst;
[[vk::binding(SceneBindings::eTextures, 1)]] Sampler2D               allTextures[];

//-----------------------------------------------------------------------
float2 GetTexcoord0(GltfRenderPrimitive renderPrim, float2 attribs)
{
    // Barycentric coordinate on the triangle
    const float3 barycentrics = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    
    // Getting the 3 indices of the triangle (local)
    uint3 triangleIndex = getTriangleIndices(renderPrim, PrimitiveIndex());
    
    // TexCoord
    return getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);
}

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true is opaque
//----------------------------------------------------------
float getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics)
{
  // Scene materials
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;  // Buffer of materials
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  if(mat.alphaMode == AlphaMode::eAlphaModeOpaque)
    return 1.0;

  // Getting the 3 indices of the triangle (local)
  uint3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

  float baseColorAlpha = 1;
  if(mat.usePbrSpecularGlossiness == 0)
  {
    baseColorAlpha = mat.pbrBaseColorFactor.a;
    if(isTexturePresent(mat.pbrBaseColorTexture))
    {
      // Retrieve the interpolated texture coordinate from the vertex
      float2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

      GltfTextureInfo texInfo = texInfos[mat.pbrBaseColorTexture];
      baseColorAlpha *= allTextures[texInfo.index].SampleLevel(uv, 0.0f).a;
    }
  }
  else
  {
    baseColorAlpha = mat.pbrDiffuseFactor.a;
    if(isTexturePresent(mat.pbrDiffuseTexture))
    {
      float2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

      GltfTextureInfo texInfo = texInfos[mat.pbrDiffuseTexture];
      baseColorAlpha *= allTextures[texInfo.index].SampleLevel(uv, 0.0f).a;
    }
  }

  baseColorAlpha *= getInterpolatedVertexColor(renderPrim, triangleIndex, barycentrics).a;

  if(mat.alphaMode == AlphaMode::eAlphaModeMask)
  {
    return baseColorAlpha >= mat.alphaCutoff ? 1.0 : 0.0;
  }

  return baseColorAlpha;
}

//-----------------------------------------------------------------------
// Pathtracer's any-hit shader deals with alpha masked materials
//-----------------------------------------------------------------------
[shader("anyhit")]
void main(inout PayloadSecondary payload, in BuiltInTriangleIntersectionAttributes attr) {

  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID   = InstanceIndex();
  uint renderPrimID = InstanceID();
  uint triangleID   = PrimitiveIndex();


  // Retrieve the Primitive mesh buffer information
  GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

  float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

  if(opacity == 0.0)
  {
    IgnoreHit();
  }
}
