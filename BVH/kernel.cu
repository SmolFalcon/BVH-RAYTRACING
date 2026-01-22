#include <GL/glew.h>
#include <GLFW/glfw3.h> 
#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <windows.h>
#include <curand_kernel.h>
#include "pfBVH.cuh"
#include "pfImage.h"



#define DLLEXPORT extern "C" __declspec(dllexport)


cudaError_t renderWithCuda(pfBVH& pBVH, pfImg& pImg, pfRay* pRays, pfVec* pLights, int pLightCount, pfMaterial* pMaterials, const pfRenderConfig& pConfig);

cudaError_t progressiveRender(float4* devAccum, pfBVH& pBVH, pfImg& pImg, pfRay* pRays, pfVec* pLights, int pLightCount, pfMaterial* pMaterials, const pfRenderConfig& pConfig, int pIter);


__device__ pfVec getLum(pfTri* pTriArr, int* pTriIdxArr, pfBVH::BVHNode* pBVHNodes, const int& pTriIdx, const pfVec& pTriBcoords, const pfVec& pLight, pfMaterial* pMaterials, const int& pSamples = 1, const float& pSoft = 0.0f)
{
    float lightIntensity = 25000;

    pfRay lR;

    lR.origin = remapPosition(pTriArr[pTriIdx],pTriBcoords) + getTriNormal(pTriArr[pTriIdx]) * 1e-2f;

    pfVec sum = { 0,0,0 };

    pfVec randVec;
    for (int s = 0; s < pSamples; s++)
    {
        //point to light
        lR.direction = pLight - lR.origin; normalize(lR.direction);
        //reset length
        lR.t = INF;

        randVec = randomHemisphereVector(lR.direction, sampleNoise(s * 2), sampleNoise(s * 1)) - lR.direction;

        lR.direction = lR.direction + randVec * pSoft;

        normalize(lR.direction);

        pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, lR);

        pfVec l = { 1,1,1 };
        if (lR.t > getNorm(lR.origin - pLight)) //no hit, add light
        {         
            sum = sum + l;
        }
        else //hit, compute shadow
        {
            invert(lR.direction);
            sum = sum + l * fminf(powf(abs(dot(lR.hit.shadingNormal, lR.direction)), 1.0f / 2.0f), 1.0f) * pMaterials[pTriArr[lR.hit.primIdx].ID].refractionWeight;
        }
    }

    sum = sum / pSamples;

    sum = (sum * lightIntensity) / powf(getNorm(lR.origin - pLight), 2.0f);

    return sum;
}


__global__ void initRaysKernel(pfRay* pRays, int pHeight, int pWidth)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {

        pfCamera cam;
        cam.origin = { 130,70,-130 };
        cam.matrix.v1 = { 0.707,0,0.707 };
        cam.matrix.v2 = { -0.159,0.974,0.159 };
        cam.matrix.v3 = { -0.689,-0.225,0.689 };

        float fovH = 45;
        float fovV = 23.4018;

        float degToRad = PI / 180;
        float tx = tanf((fovH / 2) * degToRad);
        float ty = tanf((fovV / 2) * degToRad);

        pfVec topLeft = { -tx, ty, 1 };
        pfVec sX = { 1,0,0 };
        pfVec sY = { 0,-1,0 };

        sX = sX * 2 * tx;
        sY = sY * 2 * ty;

        topLeft = cam.matrix * topLeft + cam.origin;
        sX = cam.matrix * sX;
        sY = cam.matrix * sY;


        float lx_m = (float)x / (float)pWidth;
        float ly_m = (float)y / (float)pHeight;

        pfRay lR;
        lR.origin = cam.origin;
        pfVec lTarget = topLeft + sX * lx_m + sY * ly_m;
        lR.direction = (lTarget - cam.origin); normalize(lR.direction);
        lR.hit.primIdx = -1;
        lR.isPrimary = true;


        pRays[rayIdx] = lR;

    }
}


__global__ void raytraceGIKernel(pfTri* pTriArr, int* pTriIdxArr, pfBVH::BVHNode* pBVHNodes, pfVec* pImgData, pfRay* pRays, pfVec* pLights, int plightCount, pfMaterial* pMaterials, const pfRenderConfig pConfig, int pHeight, int pWidth)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {
        pfVec camPos = pRays[rayIdx].origin;

        float shadowSoftness = pConfig.shadowSoftness;
        int shadowSamples = pConfig.primaryShadowSamples;
        int samples = 64;

        pfVec out = pMaterials[pTriArr[pRays[rayIdx].hit.primIdx].ID].color;

        for (int s = 0; s < samples; s++) //sampling
        {
            pfRay R = pRays[rayIdx]; //copy primary ray

            pfVec randVec = randomHemisphereVector(R.hit.shadingNormal, sampleNoise(s * 200), sampleNoise(s * 1)) - R.hit.shadingNormal;

            R.direction = R.hit.shadingNormal + randVec;

            normalize(R.direction);

            R.origin = R.hit.hitpos + R.hit.shadingNormal * 1e-2f;

            R.t = INF;

            pfVec f = pMaterials[pTriArr[R.hit.primIdx].ID].color;

            pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, R);

            if (R.t < INF) //hit 
            {
                for (int l = 0; l < plightCount; l++)
                {
                    //f = f + sampleBRDF(pLights[l], camPos, R.hit.hitpos, R.hit.shadingNormal, pMaterials[pTriArr[R.hit.primIdx].ID]);
                    f = f + pMaterials[pTriArr[R.hit.primIdx].ID].color * getLum(pTriArr, pTriIdxArr, pBVHNodes, R.hit.primIdx, R.hit.barycoords, pLights[l], pMaterials, shadowSamples, shadowSoftness).x;
                }

                out = out + f;
            }

        }

        if (!pRays[rayIdx].terminated)
        {
            out = (out / samples) * 255;
            out.x = (out.x > 255) ? 255 : out.x;
            out.y = (out.y > 255) ? 255 : out.y;
            out.z = (out.z > 255) ? 255 : out.z;

            pfImg::CUDA_setXY(pImgData, pHeight, x, y, out);
        }
    }

}


__global__ void raytracePrimaryKernel(pfTri* pTriArr, int* pTriIdxArr, pfBVH::BVHNode* pBVHNodes, pfVec* pImgData, pfRay* pRays, pfVec* pLights, int plightCount, pfMaterial* pMaterials, const pfRenderConfig pConfig, int pHeight, int pWidth)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {
        pfVec camPos = pRays[rayIdx].origin;

        float shadowSoftness = pConfig.shadowSoftness;
        int shadowSamples = pConfig.primaryShadowSamples;

        pfRay& R = pRays[rayIdx]; //reference the primary ray

        pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, R);

        if (R.t < INF) //hit 
        {
            R.terminated = false;
            
            pfVec out = { 0,0,0 };

            for (int l = 0; l < plightCount; l++)
            {
                out = out + sampleBRDF(pLights[l], camPos, R.hit.hitpos, R.hit.shadingNormal, pMaterials[pTriArr[R.hit.primIdx].ID]) * getLum(pTriArr, pTriIdxArr, pBVHNodes, R.hit.primIdx, R.hit.barycoords, pLights[l], pMaterials, shadowSamples, shadowSoftness).x * 255;
            }

            out.x = (out.x > 255) ? 255 : out.x;
            out.y = (out.y > 255) ? 255 : out.y;
            out.z = (out.z > 255) ? 255 : out.z;
            pfImg::CUDA_setXY(pImgData, pHeight, x, y, out);
        }
        else
        {
            R.terminated = true;
        }
    }
    
}


__global__ void raytraceReflectionKernel(pfTri* pTriArr, int* pTriIdxArr, pfBVH::BVHNode* pBVHNodes, pfVec* pImgData, const pfRay* pRays, pfVec* pLights, int plightCount, pfMaterial* pMaterials, const pfRenderConfig pConfig, int pHeight, int pWidth)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight && !pRays[rayIdx].terminated)
    {
        int reflectionDepth = pConfig.reflectionDepth;
        int roughnessSamples = pConfig.reflectionSamples;

        float shadowSoftness = pConfig.shadowSoftness;
        int shadowSamples = pConfig.secondaryShadowSamples;

        pfVec camPos = pRays[rayIdx].origin;

        pfVec sum = pfImg::CUDA_getXY(pImgData, pHeight, x, y);
        pfVec reflSum = { 0,0,0 };

        auto F = [&](const float& pIOR, const pfVec& pN, pfVec pDir) ->float //Fresnel (view normal)
            {
                float F0 = powf(pIOR - 1, 2) / powf(pIOR + 1, 2);

                float VdotN = fmaxf(dot(pDir, pN), 1e-6f);

                return F0 + (1 - F0) * powf(1 - VdotN, 5);
            };

        for (int s = 0; s < roughnessSamples; s++) //roughness sampling
        {
            pfRay R = pRays[rayIdx]; //copy primary ray

            //reflect direction around normal                   
            R.direction = pRays[rayIdx].direction - pRays[rayIdx].hit.shadingNormal * dot(pRays[rayIdx].direction, pRays[rayIdx].hit.shadingNormal) * 2;

            pfVec randVec = randomHemisphereVector(R.direction, sampleNoise(s * 200), sampleNoise(s * 1)) - R.direction;

            R.direction = R.direction + randVec * pMaterials[pTriArr[pRays[rayIdx].hit.primIdx].ID].roughness;

            normalize(R.direction);

            R.t = INF;

            R.origin = pRays[rayIdx].hit.hitpos + pRays[rayIdx].hit.geoNormal * 1e-2f;

            float fresnel = 0.0;

            fresnel += F(pMaterials[pTriArr[R.hit.primIdx].ID].IOR, R.hit.shadingNormal, R.direction);

            float reflWeight = pMaterials[pTriArr[R.hit.primIdx].ID].reflectionWeight * fminf(fresnel, 1.0);
            pfVec reflColor = pMaterials[pTriArr[R.hit.primIdx].ID].specularColor;

            for (int i = 0; i < reflectionDepth; i++) //ray depth
            {
                if (!R.terminated)
                {
                    pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, R);

                    if (R.t < INF) //hit
                    {
                        pfVec hitSample = { 0,0,0 };

                        for (int l = 0; l < plightCount; l++)
                        {
                            hitSample = hitSample + sampleBRDF(pLights[l], camPos, R.hit.hitpos, R.hit.shadingNormal, pMaterials[pTriArr[R.hit.primIdx].ID]) * getLum(pTriArr, pTriIdxArr, pBVHNodes, R.hit.primIdx, R.hit.barycoords, pLights[l], pMaterials, shadowSamples, shadowSoftness).x * 255;
                        }

                        //reflect direction around normal                   
                        R.direction = R.direction - R.hit.shadingNormal * dot(R.direction, R.hit.shadingNormal) * 2;

                        normalize(R.direction);

                        R.t = INF;

                        R.origin = R.hit.hitpos + R.hit.geoNormal * 1e-2f;

                        reflSum = reflSum + hitSample * reflWeight;

                        reflSum.x *= reflColor.x;
                        reflSum.y *= reflColor.y;
                        reflSum.z *= reflColor.z;

                        //get fresnel of hit surface
                        fresnel = 0.0;

                        fresnel += F(pMaterials[pTriArr[R.hit.primIdx].ID].IOR, R.hit.shadingNormal, R.direction);
                        fresnel = fminf(fresnel, 1.0);

                        reflWeight = reflWeight * pMaterials[pTriArr[R.hit.primIdx].ID].reflectionWeight * fresnel; //accumulate reflection weight
                        reflColor = pMaterials[pTriArr[R.hit.primIdx].ID].specularColor;

                        if (pMaterials[pTriArr[R.hit.primIdx].ID].reflectionWeight == 0) //terminate ray if hit something not reflective
                        {
                            R.terminated = true;
                        }
                    }
                    else
                    {
                        R.terminated = true;
                    }
                }
            }
        }

        sum = sum + reflSum / roughnessSamples;
        sum.x = (sum.x > 255) ? 255 : sum.x;
        sum.y = (sum.y > 255) ? 255 : sum.y;
        sum.z = (sum.z > 255) ? 255 : sum.z;

        pfImg::CUDA_setXY(pImgData, pHeight, x, y, sum);

    }

}


__global__ void raytraceRefractionKernel(pfTri* pTriArr, int* pTriIdxArr, pfBVH::BVHNode* pBVHNodes, pfVec* pImgData, pfRay* pRays, pfVec* pLights, int plightCount, pfMaterial* pMaterials, const pfRenderConfig pConfig, int pHeight, int pWidth)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {
        int refractionDepth = pConfig.refractionDepth;
        int roughnessSamples = pConfig.refractionSamples;

        float shadowSoftness = pConfig.shadowSoftness;
        int shadowSamples = pConfig.secondaryShadowSamples;

        pfVec camPos = pRays[rayIdx].origin;

        pfVec sum = pfImg::CUDA_getXY(pImgData, pHeight, x, y);
        pfVec refrSum = { 0,0,0 };

        for (int s = 0; s < roughnessSamples; s++) //roughness sampling
        {
            pfMaterial hitMat = pMaterials[pTriArr[pRays[rayIdx].hit.primIdx].ID];
            pfRay R = pRays[rayIdx]; //copy the primary ray

            for (int i = 0; i < refractionDepth; i++) //ray depth
            {
                if (!R.terminated)
                {
                    if (dot(R.hit.shadingNormal, R.direction) > 0) //refract ray direction
                    {
                        invert(R.hit.shadingNormal);
                        R.direction = snellsLaw(R.direction, R.hit.shadingNormal, hitMat.IOR, 1.0f);
                    }
                    else
                    {
                        R.direction = snellsLaw(R.direction, R.hit.shadingNormal, 1.0f, hitMat.IOR);
                    }

                    R.direction = R.direction + (randomHemisphereVector(R.direction, sampleNoise(s * 200), sampleNoise(s * 1)) - R.direction) * hitMat.roughness;

                    normalize(R.direction);

                    R.t = INF;

                    R.origin = R.hit.hitpos + R.hit.geoNormal * 1e-4 * ((dot(R.direction, R.hit.geoNormal) > 0) ? 1 : -1); //offset to be at the other side of the collision

                    pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, R);

                    if (R.t < INF)
                    {
                        pfVec out = { 0,0,0 };
                        for (int l = 0; l < plightCount; l++)
                        {
                            out = out + sampleBRDF(pLights[l], camPos, R.hit.hitpos, R.hit.shadingNormal, pMaterials[pTriArr[R.hit.primIdx].ID]) * getLum(pTriArr, pTriIdxArr, pBVHNodes, R.hit.primIdx, R.hit.barycoords, pLights[l], pMaterials, shadowSamples, shadowSoftness).x * 255;
                        }

                        refrSum = refrSum + out * hitMat.refractionWeight;

                        if (pMaterials[pTriArr[R.hit.primIdx].ID].refractionWeight < 0.1) //terminate ray if hit something not refractive
                        {
                            R.terminated = true;
                        }
                    }
                    else
                    {
                        R.terminated = true;
                    }
                }
            }
        }

        sum = sum + (refrSum / roughnessSamples);
        sum.x = (sum.x > 255) ? 255 : sum.x;
        sum.y = (sum.y > 255) ? 255 : sum.y;
        sum.z = (sum.z > 255) ? 255 : sum.z;

        pfImg::CUDA_setXY(pImgData, pHeight, x, y, sum);

    }

}


//DLL EXPORT (cache scene from Cinema 4D)
DLLEXPORT void cacheScene(pfTri* pTris, int pTriCount)
{
    pfBVH bvh(pTris, pTriCount);
        
    std::string filename = "D:/Desktop/BVH/Raytracing/bvh_cache.bin";

    bvh.BuildBVH();
    bvh.SaveBVH(filename);
}

__device__ __forceinline__ float rand01(int seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (seed & 0x00FFFFFF) / float(0x01000000);
}

__global__ void progressiveKernel(pfTri* pTriArr, int* pTriIdxArr, pfBVH::BVHNode* pBVHNodes, pfVec* pImgData, pfRay* pRays, pfVec* pLights, int plightCount, pfMaterial* pMaterials, const pfRenderConfig pConfig, int pHeight, int pWidth,int pIter)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {
        pfVec camPos = pRays[rayIdx].origin;

        pfVec sum = pfImg::CUDA_getXY(pImgData, pHeight, x, y);

        float shadowSoftness = pConfig.shadowSoftness;
        int shadowSamples = pConfig.primaryShadowSamples;

        
        pfRay R = pRays[rayIdx]; //copy primary ray

        
        pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, R); // First hit from camera

        if (R.t < INF) //First hit
        {
            pfVec out = { 0,0,0 };
            int maxBounces = 3;
            int bounces = 0;

            for (int b = 0; b < maxBounces; b++) //path tracing bounces
            {
                if (!R.terminated)
                {
                    pfMaterial hitMat = pMaterials[pTriArr[R.hit.primIdx].ID];

                    pfVec randVec = randomCosineHemisphereVector(R.hit.shadingNormal, rayIdx, b, pIter) - R.hit.shadingNormal;

                    randVec = randVec * hitMat.roughness;

                    //reflect direction around normal                   
                    //R.direction = R.direction - R.hit.shadingNormal * dot(R.direction, R.hit.shadingNormal) * 2;
					//R.direction = R.direction + randVec; normalize(R.direction);

					R.direction = randomCosineHemisphereVector(R.hit.shadingNormal, rayIdx, b, pIter); normalize(R.direction);

                    R.origin = R.hit.hitpos + R.hit.shadingNormal * 1e-2f;
                    R.t = INF;

                    pfVec lit = { 0,0,0 };
                    lit = lit + hitMat.color;


                    pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, R);

                    if (R.t < INF) // bounce hit 
                    {
                        
                        float lightIntensity = 25000;

                        for (int l = 0; l < plightCount; l++)
                        {
                            pfVec toLight = pLights[l] - R.hit.hitpos;
                            float dist2 = dot(toLight, toLight);
                            float dist = sqrt(dist2);
                            toLight = toLight / dist;

                            // Shadow ray
                            pfRay shadowRay;
                            shadowRay.origin = R.hit.hitpos + R.hit.shadingNormal * 1e-2f;
                            shadowRay.direction = toLight;
                            shadowRay.t = dist - 1e-2f;

                            pfBVH::CUDA_IntersectBVH(pTriArr, pTriIdxArr, pBVHNodes, shadowRay);
                            if (shadowRay.t >= dist - 1e-2f) // visible
                            {
                                float cosTheta = fmaxf(0.0f, dot(R.hit.shadingNormal, toLight));

                                // Lambert BRDF
                                pfVec brdf = hitMat.color * (1 / PI);

                                // Geometry + attenuation
                                pfVec contrib = brdf * cosTheta * (lightIntensity / dist2);

                                //lit = lit + contrib;
                                lit = lit + sampleBRDF(pLights[l], camPos, R.hit.hitpos, R.hit.shadingNormal, hitMat);

                            }
                        }

                        hitMat = pMaterials[pTriArr[R.hit.primIdx].ID];
                        out = out + lit;

						bounces++;
                    }
                    else // no hit
                    {
						R.terminated = true;
                    }
                }
            }

            //out = out / bounces;
            //out = out * maxBounces;
            pfImg::CUDA_setXY(pImgData, pHeight, x, y, sum + out);
        }
    }
}







__global__ void imgDataToAccum(
	const pfVec* pImgData,
	float4* accum,
	int pWidth,
	int pHeight
)
{
    //Global 2D coordinates of the thread grid (image top left is 0,0)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    int accumIdx = x + y * pWidth;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {
        pfVec c = pImgData[rayIdx];
        accum[accumIdx] = make_float4(
            float(c.x),
            float(c.y),
            float(c.z),
            1.0f);
    }
}


__global__ void tonemapKernel(
    const float4* accum,
    uchar4* output,
    int pWidth,
    int pHeight,
    int pass
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int rayIdx = y + x * pHeight;
    //only process if the thread is part of the matrix
    if (x < pWidth && y < pHeight)
    {
        float3 c;
        c.x = accum[rayIdx].x / pass;
        c.y = accum[rayIdx].y / pass;
        c.z = accum[rayIdx].z / pass;

        c.x = fminf(fmaxf(c.x, 0.0f), 1.0f);
        c.y = fminf(fmaxf(c.y, 0.0f), 1.0f);
        c.z = fminf(fmaxf(c.z, 0.0f), 1.0f);

        output[rayIdx] = make_uchar4(
            (unsigned char)(c.x * 255.0f),
            (unsigned char)(c.y * 255.0f),
            (unsigned char)(c.z * 255.0f),
            255
        );
    }
}


int main()
{
    //get the execution directory
    char cwd[MAX_PATH];
    std::string execDir;
    if (GetCurrentDirectoryA(MAX_PATH, cwd) != 0)
    {
        execDir = cwd;
    }

    int pTriCount = 1000000; //max triangles 
    pfTri* pTris = new pfTri[pTriCount];
    pfBVH bvh(pTris, pTriCount);

    //std::cout << "Input the name of the file to render: " << std::endl; //Get the scene file
    //std::string filename;  std::getline(std::cin, filename);
    std::string filename = "mainScene.bin";

    std::string imageFile = "output.png"; //define output image file

    std::cout << std::endl;

    if (std::ifstream(execDir + "\\" + filename)) //Check if the file exists
    {
        bvh.LoadBVH(execDir + "\\" + filename);
        std::cout << "Loaded " + execDir + "\\" + filename << std::endl;
        imageFile = execDir + "\\" + imageFile;
    }
    else
    {
        //Go up one level (cut the last directory \)
        size_t pos = execDir.find_last_of("\\");
        //Remove the last directory by truncating
        execDir = execDir.substr(0, pos) + "\\"; //add the slash again at the end

        if (std::ifstream(execDir + filename)) //Check again if the file exists
        {
            bvh.LoadBVH(execDir + filename);
            std::cout << "Loaded " + execDir + filename << std::endl << std::endl;
            imageFile = execDir + imageFile;
        }
        else
        {
            std::cout << "No scene file in " + execDir << std::endl;
            system("pause");
            return 0;
        }
    }


    pfImg img(500, 1000);
    pfRay* rays = new pfRay[img.height * img.width];

    int lightCount = 2;
    pfVec* Lights = new pfVec[lightCount];
    Lights[0] = { 65,135,-120 };
    Lights[1] = { 65,135,120 };
    //Lights[0] = { 0,90,0 }; Lights[1] = { 130,70,-130 };//shadowBall lights


    pfMaterial* Mat = new pfMaterial[6];
    ///*
    //Teapot
    Mat[0].color = { 0.7,1,1 }; Mat[0].reflectionWeight = 0.0; Mat[0].IOR = 1.5; Mat[0].roughness = 0.5;
    //Mirrors
    Mat[1].color = { 0,0,0 }; Mat[1].reflectionWeight = 1.0; Mat[1].IOR = 0.0; Mat[1].roughness = 0.01;
    //Left ball
    Mat[2].color = { 0,0,0 }; Mat[2].reflectionWeight = 1.0; Mat[2].refractionWeight = 1.0; Mat[2].roughness = 0.005; Mat[2].IOR = 2.5;
    //Floor
	Mat[3].color = { 0.5,0.5,0.5 }; Mat[3].roughness = 0.8;
    //Skull
    Mat[4].color = { 1,0,0 }; Mat[4].reflectionWeight = 2.0; Mat[4].specularColor = { 1,0.8,0 }; Mat[4].IOR = 0.025; Mat[4].roughness = 0.15;
    //Right ball
    Mat[5].color = { 0,0.35,1 }; Mat[5].reflectionWeight = 1.0; Mat[5].IOR = 0.1; Mat[5].roughness = 0.05;
    //*/

    //Set render sampling parameters
    pfRenderConfig rConfig; //rConfig.shadowSoftness = 0.03; rConfig.primaryShadowSamples = 64; //enable soft shadows
    rConfig.refractionSamples = 16; rConfig.refractionDepth = 16;








	// -------------------- IPR with OpenGL --------------------


    // -----------------------------
    // Init GLFW
    // -----------------------------
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        img.width, img.height,
        "IPR",
        nullptr, nullptr
    );

    glfwMakeContextCurrent(window);

    // -----------------------------
    // Init GLEW
    // -----------------------------
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to init GLEW\n";
        return -1;
    }

    // IMPORTANT: CUDA device AFTER GL context exists
    cudaSetDevice(0);
    float4* devAccum = nullptr;
    cudaMalloc(&devAccum, img.width * img.height * sizeof(float4));
    cudaMemset(devAccum, 0, img.width * img.height * sizeof(float4));

    // -----------------------------
    // Create texture
    // -----------------------------
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        img.width, img.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // -----------------------------
    // Create PBO
    // -----------------------------
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        img.width * img.height * sizeof(uchar4),
        nullptr,
        GL_DYNAMIC_DRAW
    );

    // Register PBO with CUDA
    cudaGraphicsResource* cudaPBO;
    cudaGraphicsGLRegisterBuffer(
        &cudaPBO,
        pbo,
        cudaGraphicsRegisterFlagsWriteDiscard
    );

    // -----------------------------
    // Render loop
    // -----------------------------
    dim3 threads(16, 16);
    dim3 blocks(
        (img.width + threads.x - 1) / threads.x,
        (img.height + threads.y - 1) / threads.y
    );

    static int pass = 0;
    while (!glfwWindowShouldClose(window))
    {
        pass++;

        uchar4* devOut = nullptr;
        size_t size = 0;

        // Map PBO
        cudaGraphicsMapResources(1, &cudaPBO);
        cudaGraphicsResourceGetMappedPointer((void**)&devOut, &size, cudaPBO);

        // Accumulate one noisy sample
        //accumulateKernel << <blocks, threads >> > (devAccum, img.width, img.height, pass);


        auto start = std::chrono::high_resolution_clock::now();

        cudaError_t cudaStatus = progressiveRender(devAccum, bvh, img, rays, Lights, lightCount, Mat, rConfig,pass);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
        std::cout << std::endl << "Rendered image in: " << duration.count() / 1000.0 << " seconds" << std::endl << std::endl;



        // Tonemap accumulated result
        tonemapKernel << <blocks, threads >> > (devAccum, devOut, img.width, img.height, pass);



        cudaDeviceSynchronize();

        // Unmap PBO
        cudaGraphicsUnmapResources(1, &cudaPBO);

        // Upload PBO → texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.width, img.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // Draw fullscreen quad (compatibility profile)
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);

        glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, 1);
        glTexCoord2f(0, 0); glVertex2f(-1, 1);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // -----------------------------
    // Cleanup
    // -----------------------------
    delete[] rays;
    delete[] pTris;
    delete[] Lights;
    delete[] Mat;
    cudaFree(devAccum);
    cudaGraphicsUnregisterResource(cudaPBO);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}



cudaError_t progressiveRender(float4* devAccum, pfBVH& pBVH, pfImg& pImg, pfRay* pRays, pfVec* pLights, int pLightCount, pfMaterial* pMaterials, const pfRenderConfig& pConfig, int pIter)
{
    pfTri* dev_tris;
    int* dev_triIdx;
    pfBVH::BVHNode* dev_bvhNodes;
    pfVec* dev_imgData;
    pfRay* dev_rays;
    pfVec* dev_lights;
    pfMaterial* dev_materials;
    //pBVH.triIdx[0] = pBVH.N;


    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    //Store triangles
    cudaStatus = cudaMalloc(&dev_tris, sizeof(pfTri) * pBVH.N);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_tris, pBVH.tri, sizeof(pfTri) * pBVH.N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Store triangle indices from BVH structure
    cudaStatus = cudaMalloc(&dev_triIdx, sizeof(int) * pBVH.N);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_triIdx, pBVH.triIdx, sizeof(int) * pBVH.N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Store BVH nodes
    cudaStatus = cudaMalloc(&dev_bvhNodes, sizeof(pfBVH::BVHNode) * pBVH.nodesUsed);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_bvhNodes, pBVH.bvhNode, sizeof(pfBVH::BVHNode) * pBVH.nodesUsed, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Image memory allocation
    cudaStatus = cudaMalloc(&dev_imgData, sizeof(pfVec) * pImg.height * pImg.width);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_imgData, pImg.data, sizeof(pfVec) * pImg.height * pImg.width, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Rays memory allocation
    cudaStatus = cudaMalloc(&dev_rays, sizeof(pfRay) * pImg.height * pImg.width);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_rays, pRays, sizeof(pfRay) * pImg.height * pImg.width, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Lights memory allocation
    cudaStatus = cudaMalloc(&dev_lights, sizeof(pfVec) * pLightCount);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_lights, pLights, sizeof(pfVec) * pLightCount, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;



    //Materials memory allocation
    cudaStatus = cudaMalloc(&dev_materials, sizeof(pfMaterial) * 6);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_materials, pMaterials, sizeof(pfMaterial) * 6, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;




    int xThreads = 16;
    int yThreads = 16;

    int xBlocks = (pImg.width + yThreads - 1) / xThreads;
    int yBlocks = (pImg.height + yThreads - 1) / yThreads;



    //initialize camera rays
    initRaysKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_rays, pImg.height, pImg.width); std::cout << "Rays initialized" << std::endl;

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) goto Error;


    progressiveKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width, pIter); std::cout << "Primary rays... ";

	imgDataToAccum << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_imgData, devAccum, pImg.width, pImg.height);

    cudaStatus = cudaDeviceSynchronize(); std::cout << "done" << std::endl;
    if (cudaStatus != cudaSuccess) goto Error;
    /*
    raytraceReflectionKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width); std::cout << "Reflection rays... ";

    cudaStatus = cudaDeviceSynchronize(); std::cout << "done" << std::endl;
    if (cudaStatus != cudaSuccess) goto Error;

    raytraceRefractionKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width); std::cout << "Refraction rays... ";

    cudaStatus = cudaDeviceSynchronize(); std::cout << "done" << std::endl;
    if (cudaStatus != cudaSuccess) goto Error;
    */

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) goto Error;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) goto Error;


    cudaStatus = cudaMemcpy(pImg.data, dev_imgData, sizeof(pfVec) * pImg.height * pImg.width, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) goto Error;

Error:
    cudaFree(dev_tris);
    cudaFree(dev_triIdx);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_imgData);
    cudaFree(dev_rays);
    cudaFree(dev_lights);
    cudaFree(dev_materials);

    return cudaStatus;
}













int BUmain()
{
    //get the execution directory
    char cwd[MAX_PATH];
    std::string execDir;
    if (GetCurrentDirectoryA(MAX_PATH, cwd) != 0)
    {
        execDir = cwd;
    }

    int pTriCount = 1000000; //max triangles 
    pfTri* pTris = new pfTri[pTriCount];
    pfBVH bvh(pTris, pTriCount);

    std::cout << "Input the name of the file to render: "<<std::endl; //Get the scene file
    std::string filename;  std::getline(std::cin, filename);

    std::string imageFile = "output.png"; //define output image file

    std::cout << std::endl;

    if (std::ifstream(execDir + "\\" + filename)) //Check if the file exists
    {
        bvh.LoadBVH(execDir + "\\" + filename);
        std::cout << "Loaded " + execDir + "\\" + filename << std::endl;
        imageFile = execDir + "\\" + imageFile;
    }
    else
    {
        //Go up one level (cut the last directory \)
        size_t pos = execDir.find_last_of("\\");
        //Remove the last directory by truncating
        execDir = execDir.substr(0, pos) + "\\"; //add the slash again at the end

        if (std::ifstream(execDir + filename)) //Check again if the file exists
        {
            bvh.LoadBVH(execDir + filename);
            std::cout << "Loaded " +execDir + filename << std::endl << std::endl;
            imageFile = execDir + imageFile;
        }
        else
        {
            std::cout << "No scene file in " + execDir << std::endl;
            system("pause");
            return 0;
        }
    }


    pfImg img(1024, 2048);
    pfRay* rays = new pfRay[img.height * img.width];

    int lightCount = 2;
    pfVec* Lights = new pfVec[lightCount];
    Lights[0] = {65,135,-120};
    Lights[1] = {65,135,120};
    //Lights[0] = { 0,90,0 }; Lights[1] = { 130,70,-130 };//shadowBall lights

    
    pfMaterial* Mat = new pfMaterial[6];
    ///*
    //Teapot
    Mat[0].color = { 0.7,1,1 }; Mat[0].reflectionWeight = 0.0; Mat[0].IOR = 1.5; Mat[0].roughness = 0.5;
    //Mirrors
    Mat[1].color = { 0,0,0 }; Mat[1].reflectionWeight = 1.0; Mat[1].IOR = 0.0; Mat[1].roughness = 0.01;
    //Left ball
    Mat[2].color = { 0,0,0 }; Mat[2].reflectionWeight = 1.0; Mat[2].refractionWeight = 1.0; Mat[2].roughness = 0.005; Mat[2].IOR = 2.5;
    //Floor
    Mat[3].color = { 0.5,0.5,0.5 };
    //Skull
    Mat[4].color = { 1,0,0 }; Mat[4].reflectionWeight = 2.0; Mat[4].specularColor = { 1,0.8,0 }; Mat[4].IOR = 0.025; Mat[4].roughness = 0.15;
    //Right ball
    Mat[5].color = { 0,0.35,1 }; Mat[5].reflectionWeight = 1.0; Mat[5].IOR = 0.1; Mat[5].roughness = 0.05;
    //*/

    //Set render sampling parameters
    pfRenderConfig rConfig; //rConfig.shadowSoftness = 0.03; rConfig.primaryShadowSamples = 64; //enable soft shadows
    rConfig.refractionSamples = 16; rConfig.refractionDepth = 16;

    auto start = std::chrono::high_resolution_clock::now();

    cudaError_t cudaStatus = renderWithCuda(bvh, img, rays, Lights, lightCount, Mat, rConfig);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
    std::cout << std::endl << "Rendered image in: " << duration.count() / 1000.0 << " seconds" << std::endl << std::endl;

    cudaStatus = cudaDeviceReset();


    delete[] rays;
    delete[] pTris;
    delete[] Lights;
    delete[] Mat;


    //img.normalizeImage();
    img.storeImage(imageFile.c_str());

    system("pause");
    return 0;
}

cudaError_t renderWithCuda(pfBVH& pBVH, pfImg& pImg, pfRay* pRays, pfVec* pLights, int pLightCount, pfMaterial* pMaterials, const pfRenderConfig& pConfig)
{
    pfTri* dev_tris;
    int* dev_triIdx;
    pfBVH::BVHNode* dev_bvhNodes;
    pfVec* dev_imgData;
    pfRay* dev_rays;
    pfVec* dev_lights;
    pfMaterial* dev_materials;
    //pBVH.triIdx[0] = pBVH.N;


    cudaError_t cudaStatus;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    //Store triangles
    cudaStatus = cudaMalloc(&dev_tris, sizeof(pfTri) * pBVH.N);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_tris, pBVH.tri, sizeof(pfTri) * pBVH.N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Store triangle indices from BVH structure
    cudaStatus = cudaMalloc(&dev_triIdx, sizeof(int) * pBVH.N);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_triIdx, pBVH.triIdx, sizeof(int) * pBVH.N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Store BVH nodes
    cudaStatus = cudaMalloc(&dev_bvhNodes, sizeof(pfBVH::BVHNode) * pBVH.nodesUsed);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_bvhNodes, pBVH.bvhNode, sizeof(pfBVH::BVHNode) * pBVH.nodesUsed, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Image memory allocation
    cudaStatus = cudaMalloc(&dev_imgData, sizeof(pfVec) * pImg.height * pImg.width);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_imgData, pImg.data, sizeof(pfVec) * pImg.height * pImg.width, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Rays memory allocation
    cudaStatus = cudaMalloc(&dev_rays, sizeof(pfRay) * pImg.height * pImg.width);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_rays, pRays, sizeof(pfRay) * pImg.height * pImg.width, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;


    //Lights memory allocation
    cudaStatus = cudaMalloc(&dev_lights, sizeof(pfVec) * pLightCount);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_lights, pLights, sizeof(pfVec) * pLightCount, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;



    //Materials memory allocation
    cudaStatus = cudaMalloc(&dev_materials, sizeof(pfMaterial) * 6);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_materials, pMaterials, sizeof(pfMaterial) * 6, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;




    int xThreads = 16;
    int yThreads = 16;

    int xBlocks = (pImg.width + yThreads - 1) / xThreads;
    int yBlocks = (pImg.height + yThreads - 1) / yThreads;



    //initialize camera rays
    initRaysKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_rays, pImg.height, pImg.width); std::cout << "Rays initialized" << std::endl;
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) goto Error;

    
    raytracePrimaryKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width); std::cout << "Primary rays... ";
    
    cudaStatus = cudaDeviceSynchronize(); std::cout<< "done" << std::endl;
    if (cudaStatus != cudaSuccess) goto Error;
    ///*
    raytraceReflectionKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width); std::cout << "Reflection rays... ";

    cudaStatus = cudaDeviceSynchronize(); std::cout << "done" << std::endl;
    if (cudaStatus != cudaSuccess) goto Error;

    raytraceRefractionKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width); std::cout << "Refraction rays... ";

    cudaStatus = cudaDeviceSynchronize(); std::cout << "done" << std::endl;
    if (cudaStatus != cudaSuccess) goto Error;
    //*/

    //raytraceGIKernel << < dim3(xBlocks, yBlocks), dim3(xThreads, yThreads) >> > (dev_tris, dev_triIdx, dev_bvhNodes, dev_imgData, dev_rays, dev_lights, pLightCount, dev_materials, pConfig, pImg.height, pImg.width); std::cout << "GI rays... ";

    //cudaStatus = cudaDeviceSynchronize(); std::cout << "done" << std::endl;
    //if (cudaStatus != cudaSuccess) goto Error;

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) goto Error;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) goto Error;


    cudaStatus = cudaMemcpy(pImg.data, dev_imgData, sizeof(pfVec) * pImg.height * pImg.width, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) goto Error;

Error:
    cudaFree(dev_tris);
    cudaFree(dev_triIdx);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_imgData);
    cudaFree(dev_rays);
    cudaFree(dev_lights);
    cudaFree(dev_materials);

    return cudaStatus;
}
