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
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/ray_utils.h.slang"

#include "dlss_helper.slang"

// Individual binding points
[[vk::binding(RtxBindings::eTlas, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(SceneBindings::eTextures, 1)]] Sampler2D texturesMap[];

[[vk::binding(DlssBindings::eColor, 2)]] RWTexture2D<float4> dlssColor;
[[vk::binding(DlssBindings::eMotionVectors, 2)]] RWTexture2D<float4> dlssObjectMotion;
[[vk::binding(DlssBindings::eNormal_Roughness, 2)]] RWTexture2D<float4> dlssNormalRoughness;
[[vk::binding(DlssBindings::eViewZ, 2)]] RWTexture2D<float4> dlssViewZ;
[[vk::binding(DlssBindings::eSpecAlbedo, 2)]] RWTexture2D<float4> dlssSpecAlbedo;
[[vk::binding(DlssBindings::eSpecHitDist, 2)]] RWTexture2D<float4> dlssSpecHitDistance;
[[vk::binding(DlssBindings::eBaseColor_Metalness, 2)]] RWTexture2D<float4> dlssBaseColorMetalness;

[[vk::binding(EnvBindings::eImpSamples, 3)]] StructuredBuffer<EnvAccel> envSamplingData;
[[vk::binding(EnvBindings::eHdr, 3)]] Sampler2D hdrTexture;

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pc;

struct HitState
{
    float3 pos;
    float3 nrm;
    float3 geonrm;
    float2 uv;
    float3 tangent;
    float3 bitangent;
    float bitangentSign;
};

static float2 maxRoughness = float2(0.0);

//-----------------------------------------------------------------------
// Direct contribution from all lights (no HDR environment)
//-----------------------------------------------------------------------
float3 DirectLight(PbrMaterial matEval, HitState hitState, float3 toEye)
{
    float3 contribRadiance = float3(0);
#if NB_LIGHTS > 0
    uint nbLight = NB_LIGHTS;

    for(int light_index = 0; light_index < nbLight; light_index++)
    {
        Light light = frameInfo.light[light_index];

        float3 lightDir;
        float3 lightContrib = lightContribution(light, hitState.pos, hitState.nrm, lightDir);
        float lightDist = (light.type != 0) ? 1e37f : length(hitState.pos - light.position);
        float dotNL = dot(lightDir, hitState.nrm);

        if(dotNL > 0.0)
        {
            float lightPdf = 1.0f / float(NB_LIGHTS);

            float pdf = 0;
            float3 brdf = pbrEval(matEval, toEye, lightDir, pdf);
            float3 radiance = brdf * dotNL * lightContrib / lightPdf;

            // Shadow ray
            RayDesc ray;
            ray.Origin = hitState.pos;
            ray.Direction = lightDir;
            ray.TMin = 0.001;
            ray.TMax = lightDist;

            PayloadSecondary payload;
            payload.hitT = 0;
            
            TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                     0xFF, SBTOFFSET_SECONDARY, 0, MISSINDEX_SECONDARY, ray, payload);

            // If hitting nothing, add light contribution
            if(abs(payload.hitT) == DLSS_INF_DISTANCE)
            {
                contribRadiance += radiance;
            }
        }
    }
#endif

    return contribRadiance;
}

//-----------------------------------------------------------------------
// Direct contribution of the HDR environment
//-----------------------------------------------------------------------
void HdrContrib(PbrMaterial pbrMat, float3 startPos, float3 toEye, out float3 outRadiance, inout PayloadSecondary payload)
{
    outRadiance = float3(0);

    float3 lightDir;
    float3 lightContrib;
    float lightPdf;

    float3 randVal = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));

    if(TEST_FLAG(pc.frameInfo->flags, FLAGS_ENVMAP_SKY))
    {
        SkySamplingResult skySample = samplePhysicalSky(*pc.skyParams, randVal.xy);
        lightDir = skySample.direction;
        lightContrib = skySample.radiance;
        lightPdf = skySample.pdf;
    }
    else
    {
        // Sample envmap in random direction
        float4 radiance_pdf = environmentSample(hdrTexture, envSamplingData, randVal, lightDir);
        // rotate returned direction into worldspace
        lightDir = rotate(lightDir, float3(0, 1, 0), pc.frameInfo->envRotation);

        lightContrib = radiance_pdf.xyz;
        lightPdf = radiance_pdf.w;
    }

    lightContrib *= pc.frameInfo->envIntensity.xyz;

    float dotNL = dot(lightDir, pbrMat.N);

    // above surface?
    if(dotNL > 0.0 && lightPdf > 0.0)
    {
        float3 radiance;

        BsdfEvaluateData bsdfEval;
        bsdfEval.k1 = toEye;
        bsdfEval.k2 = lightDir;
        bsdfEval.xi = randVal;

        bsdfEvaluate(bsdfEval, pbrMat);

        if(bsdfEval.pdf > 0.0)
        {
            const float mis_weight = powerHeuristic(lightPdf, bsdfEval.pdf);

            float3 lightRadiance = mis_weight * lightContrib / lightPdf;
            radiance = (bsdfEval.bsdf_diffuse + bsdfEval.bsdf_glossy) * lightRadiance;

            // Shadow ray
            RayDesc ray;
            ray.Origin = startPos;
            ray.Direction = lightDir;
            ray.TMin = 0.001;
            ray.TMax = DLSS_INF_DISTANCE;

            payload.hitT = 0;
            TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                     0xFF, SBTOFFSET_SECONDARY, 0, MISSINDEX_SECONDARY, ray, payload);

            // If ray to sky is not blocked, this is the environment light contribution
            if(abs(payload.hitT) == DLSS_INF_DISTANCE)
            {
                outRadiance = radiance;
            }
        }
    }
}

//-----------------------------------------------------------------------
// Build Hit information from the payload's returned data and evaluate the
// material at the hit position
//-----------------------------------------------------------------------
void buildHitInfo(PayloadPrimary payloadPrimary, float3 rayOrigin, float3 rayDirection, out PbrMaterial pbrMat, out HitState hitState)
{
    // Retrieve the Primitive mesh buffer information
    GltfRenderNode GltfRenderNode = pc.gltfScene->renderNodes[payloadPrimary.renderNodeIndex];
    GltfRenderPrimitive renderPrim = pc.gltfScene->renderPrimitives[payloadPrimary.renderPrimIndex];

    // Calculate hitState position, normal tangent etc
    hitState.pos = rayOrigin + payloadPrimary.hitT * rayDirection;
    hitState.nrm = payloadPrimary.normal_envmapRadiance;
    hitState.geonrm = hitState.nrm;
    hitState.uv = payloadPrimary.uv;
    hitState.tangent = payloadPrimary.tangent;
    hitState.bitangent = cross(hitState.nrm, hitState.tangent) * payloadPrimary.bitangentSign;

    // Scene materials
    uint matIndex = max(0, GltfRenderNode.materialID);
    GltfShadeMaterial mat = pc.gltfScene->materials[matIndex];
    GltfTextureInfo *texInfos = pc.gltfScene->textureInfos;
    pbrMat = evaluateMaterial(mat, hitState.nrm, hitState.tangent, hitState.bitangent, hitState.uv, texturesMap, texInfos);

    if(pc.overrideRoughness > 0)
    {
        pbrMat.roughness = float2(clamp(pc.overrideRoughness, MICROFACET_MIN_ROUGHNESS, 1.0));
        pbrMat.roughness *= pbrMat.roughness;
    }
    if(pc.overrideMetallic > 0)
    {
        pbrMat.metallic = pc.overrideMetallic;
    }

    if(TEST_FLAG(pc.frameInfo->flags, FLAGS_USE_PATH_REGULARIZATION))
    {
        maxRoughness = max(maxRoughness, pbrMat.roughness);
        pbrMat.roughness = maxRoughness;
    }
}

// Motion vector computation
float2 computeCameraMotionVector(float2 pixelCenter, float4 motionOrigin)
{
    float4 oldPos = mul(motionOrigin, pc.frameInfo->prevMVP);
    oldPos.xy /= oldPos.w;
    oldPos.xy = (oldPos.xy * 0.5 + 0.5) * float2(DispatchRaysDimensions().xy);
    float2 motionVec = oldPos.xy - pixelCenter.xy;
    return motionVec;
}

//-----------------------------------------------------------------------
// ENTRY function
//-----------------------------------------------------------------------
[shader("raygeneration")]
void main()
{
    int2 pixelPos = int2(DispatchRaysIndex().xy);
    
    // Initialize the random number
    PayloadSecondary payload;
    payload.seed = xxhash32(uint3(DispatchRaysIndex().xy, pc.frame));
    
    float2 pixelCenter = float2(DispatchRaysIndex().xy) + 0.5;
    float2 unjitteredPixelCenter = pixelCenter;
    
    pixelCenter += pc.frameInfo->jitter;
    
    const float2 inUV = pixelCenter / float2(DispatchRaysDimensions().xy);
    const float2 d = inUV * 2.0 - 1.0;
    float3 origin = mul(float4(0.0, 0.0, 0.0, 1.0), pc.frameInfo->viewInv).xyz;
    const float3 eyePos = origin.xyz;
    const float4 target = mul(float4(d.x, d.y, 0.01, 1.0), pc.frameInfo->projInv);
    float3 direction = normalize(mul( target.xyz, (float3x3)pc.frameInfo->viewInv));
    const float3 orgDirection = direction;
    float3 toEye = -direction.xyz;
    const uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    
    PbrMaterial pbrMat;  // Material at hit position
    HitState hitState;
    
    // Result of trace
    bool hitSky = false;
    bool isPsr = false;
    float psrHitDist = 0.0;
    float3 psrThroughput = float3(1.0);
    float3 psrDirectRadiance = float3(0.0);
    float3x3 psrMirror = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);  // identity
    
    //====================================================================================================================
    // STEP 1 - Find first non-mirror primary hit.
    // The first non-mirror hit surface is used as 'Primary Surface Replacement'.
    // Collect G-Buffer material & hit information.
    // #PSR
    //====================================================================================================================
    int psrDepth = 0;
    PayloadPrimary payloadPrimary;
    
    do
    {
        payloadPrimary.hitT = 0;
        
        // Trace using primary closest hit, miss shaders and PayloadPrimary
        RayDesc ray;
        ray.Origin = origin.xyz;
        ray.Direction = direction.xyz;
        ray.TMin = 0.01;
        ray.TMax = 1e32;
        
        TraceRay(topLevelAS, rayFlags, 0xFF, SBTOFFSET_PRIMARY, 0, MISSINDEX_PRIMARY, ray, payloadPrimary);
        
        hitSky = (payloadPrimary.hitT == DLSS_INF_DISTANCE);
        if(hitSky)
        {
            psrDirectRadiance += psrThroughput * payloadPrimary.normal_envmapRadiance;
            break;
        }
        
        // Accumulate the hit distances along the mirrored reflections - used to calculate the
        // virtual world PSR position's ViewZ distance
        psrHitDist += payloadPrimary.hitT;
        
        buildHitInfo(payloadPrimary, origin, direction, pbrMat, hitState);
        origin = offsetRay(hitState.pos, pbrMat.Ng);
        
        // Did we hit anything other than a mirror?
        if((pbrMat.roughness.x > ((MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS) + 0.001)) || pbrMat.metallic < 1.0
           || !TEST_FLAG(pc.frameInfo->flags, FLAGS_USE_PSR))
        {
            break;
        }
        
        // At least one mirror hit
        isPsr = true;
        
        // Only the glossy part should be non-zero as this is a mirror surface.
        // The pdf for the mirrored reflection should be infinity
        psrDirectRadiance += psrThroughput * pbrMat.emissive;
        
        {
            BsdfSampleData specBsdfSample;
            specBsdfSample.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
            specBsdfSample.k1 = -direction;
            
            bsdfSample(specBsdfSample, pbrMat);
            
            if(specBsdfSample.event_type != BSDF_EVENT_GLOSSY_REFLECTION)
            {
                // Debug: should not be possible as this is a mirror surface
                pbrMat.baseColor = float3(10.0, 0.0, 10.0);
                break;
            }
            
            psrThroughput *= specBsdfSample.bsdf_over_pdf;
            psrMirror = mul(psrMirror, buildMirrorMatrix(pbrMat.N));
            
            direction = specBsdfSample.k2;
        }
        
        ++psrDepth;
    } while(psrDepth < 5);
    
    float3 virtualOrigin = eyePos + orgDirection * psrHitDist;
    float viewDepth = -mul(pc.frameInfo->view, float4(virtualOrigin, 1.0)).z;  // NOTE: viewZ is the 'Z' of the world hit position in camera space
    
    // Early out when hitting sky (even via mirrors)
    if(hitSky)
    {
        dlssColor[pixelPos] = float4(psrDirectRadiance, 1.0);
        dlssSpecAlbedo[pixelPos] = float4(0);
        // We can't store the HDR sky color as diffuse albedo.
        // Use a tonemapped version of the sky color as demodulation guide for DLSS_RR,
        // so it'll hopefully preserve detail in the sky and not attempt to denoise it.
        dlssBaseColorMetalness[pixelPos] = float4(reinhardMax(psrDirectRadiance), pbrMat.metallic);
        dlssNormalRoughness[pixelPos] = float4(0);
        dlssSpecHitDistance[pixelPos] = float4(0.0);
        
        float4 motionOrigin;
        if (!isPsr)
        {
            // This is the case when we hit the skybox directly.
            // Treat it like a point at infinity along the view vector
            dlssViewZ[pixelPos] = float4(DLSS_INF_DISTANCE);
            // motion origin is point at infinity along view vector
            motionOrigin = float4(orgDirection, 0.0);
        }
        else
        {
            // Here we hit the sky through a reflection on the primary surface
            // Pretend the sky is "texture mapped" onto the mirror and moves with it.
            dlssViewZ[pixelPos] = float4(viewDepth);
            motionOrigin = float4(virtualOrigin, 1.0);
        }
        
        float2 motionVec = computeCameraMotionVector(pixelCenter, motionOrigin);
        dlssObjectMotion[pixelPos] = float4(motionVec, float2(0.0));
        return;
    }
    
    // ViewZ buffer
    dlssViewZ[pixelPos] = float4(viewDepth);
    
    // Normal/Roughness buffer
    {
        // Transform surface normal from "virtual world normal" to world normal through a series of mirror-matrix.
        // In case of NOT hitting any mirror, 'psrMirror' is just the identity matrix
        float3 worldNormal = mul(psrMirror, pbrMat.N);
        
        float4 normalRoughness = float4(worldNormal, sqrt(pbrMat.roughness.x));
        dlssNormalRoughness[pixelPos] = normalRoughness;
    }
    
    // Tint the material by the accumulated tinting of the mirrors until we reached the PSR
    // 'psrThroughput' will be (1.0, 1.0, 1.0) when hitting no mirrors.
    pbrMat.baseColor *= psrThroughput;
    pbrMat.specularColor *= psrThroughput;
    pbrMat.emissive = pbrMat.emissive * psrThroughput + psrDirectRadiance;
    
    // Motion Vector Buffer
    {
        float2 motionVec = computeCameraMotionVector(pixelCenter, float4(virtualOrigin, 1.0));
        dlssObjectMotion[pixelPos] = float4(motionVec, 0.0, 0.0);
    }
    
    // transform eye vector into "virtual world" for PSR surfaces (identity if primary hit is non-mirror material)
    // -direction happens to be the same direction as if we did 'toEye = toEye * psrMirror;'
    toEye = -direction;
    
    {
        // BaseColor/Metalness Buffer. DLSS only needs the base color ("Diffuse Albedo")
        dlssBaseColorMetalness[pixelPos] = float4(pbrMat.baseColor, pbrMat.metallic);
    }
    
    //====================================================================================================================
    // STEP 2 - Get the direct light contribution at hit position
    //====================================================================================================================
    
    // Getting contribution of HDR
    float3 hdrRadiance = float3(0);
    
    HdrContrib(pbrMat, hitState.pos, toEye, hdrRadiance, payload);
    
    // Contribution of all lights
    float3 directLum = DirectLight(pbrMat, hitState, toEye);
    
    directLum += psrDirectRadiance + pbrMat.emissive;
    
    //====================================================================================================================
    // STEP 3 - Get the indirect contribution at hit position
    //====================================================================================================================
    
    float3 radiance = hdrRadiance;
    
    float pathLength = 0.0;  // if first hit creates absorption event, provide a hitdist of 0
    
    //====================================================================================================================
    // STEP 3.1 - Sampling direction
    //====================================================================================================================
    
    BsdfSampleData sampleData;
    sampleData.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
    sampleData.k1 = toEye;
    bsdfSample(sampleData, pbrMat);
    
    if(sampleData.event_type != BSDF_EVENT_ABSORB)
    {
        //====================================================================================================================
        // STEP 3.2 - Evaluation of throughput for the hit outgoing direction
        //====================================================================================================================
        
        // Resetting payload
        payload.contrib = float3(0.0);
        payload.weight = float3(1.0);
        payload.hitT = DLSS_INF_DISTANCE;
        payload.rayDirection = sampleData.k2;
        payload.rayOrigin = origin;
        payload.bsdfPDF = sampleData.pdf;
        payload.maxRoughness = maxRoughness;
        
        //====================================================================================================================
        // STEP 3.3 - Trace ray from depth 1 and path trace until the ray dies
        //====================================================================================================================
        float3 throughput = sampleData.bsdf_over_pdf;
        
        for(int depth = 1; depth < pc.maxDepth; depth++)
        {
            payload.hitT = DLSS_INF_DISTANCE;
            
            RayDesc secondaryRay;
            secondaryRay.Origin = payload.rayOrigin;
            secondaryRay.Direction = payload.rayDirection;
            secondaryRay.TMin = 0.001;
            secondaryRay.TMax = DLSS_INF_DISTANCE;
            
            TraceRay(topLevelAS, rayFlags, 0xFF, SBTOFFSET_SECONDARY, 0, MISSINDEX_SECONDARY, secondaryRay, payload);
            
            // Accumulating results
            radiance += payload.contrib * throughput;
            throughput *= payload.weight;
            
            // The first secondary path segment determines the specular hit distance.
            // If the ray hits the environment, -DLSS_INF_DISTANCE is returned
            if(depth == 1 && sampleData.event_type == BSDF_EVENT_GLOSSY_REFLECTION)
            {
                pathLength = abs(payload.hitT);
            }
            
            if(payload.hitT < 0.0)
            {
                break;
            }
        }
        
        // Removing fireflies
        // float lum = dot(radiance, float3(0.212671f, 0.715160f, 0.072169f));
        // if(lum > pc.maxLuminance)
        // {
        //   radiance *= pc.maxLuminance / lum;
        // }
    }
    
    // Environment ( pre-integrated ) specular term
    float3 Fenv = float3(0.0);
    if(sampleData.event_type != BSDF_EVENT_DIFFUSE)
    {
        float VdotN = dot(toEye, pbrMat.N);
        Fenv = EnvironmentTerm_Rtg(pbrMat.specularColor, max(VdotN, 0.0), pbrMat.roughness.x);
    }
    
    // Store final specular color at pixel
    dlssSpecAlbedo[pixelPos] = float4(Fenv, 0.0);
    dlssSpecHitDistance[pixelPos] = float4(pathLength);
    
    dlssColor[pixelPos] = float4(radiance + directLum, pbrMat.opacity);
} 
