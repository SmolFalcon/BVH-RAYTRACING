#include "cuda_runtime.h"
#include "cuda_noise.cuh"
#include <stdio.h>

#define PI 3.14159265358979323f
#define INF 1e30f
#define uint unsigned int


struct pfVec
{
    float x = 0;
    float y = 0;
    float z = 0;

    //overload operators, added __host__ __device__ to make them usable anywhere

    __host__ __device__ float operator[](const int& index) const 
    {
        if (index == 0) return x;
        if (index == 1) return y;
        if (index == 2) return z;
    }

    __host__ __device__ pfVec operator+(const pfVec& V) const 
    {
        return { x + V.x ,y + V.y ,z + V.z };
    }

    __host__ __device__ pfVec operator-(const pfVec& V) const 
    {
        return { x - V.x ,y - V.y ,z - V.z };
    }

    __host__ __device__ pfVec operator*(const float& V) const
    {
        return { x * V ,y * V ,z * V };
    }

    __host__ __device__ pfVec operator/(const float& V) const
    {
        return { x / V ,y / V ,z / V };
    }
};

struct pfMaterial
{
    float IOR = 1.5;
    pfVec color = { 1,1,1 };
    pfVec specularColor = { 1,1,1 };
    float diffuseRoughness = 0;
    float roughness = 0;
    float reflectionWeight = 0;
    float refractionWeight = 0;
    pfVec emission = { 0,0,0 };
};

struct pfRenderConfig
{
    int primaryShadowSamples = 1;
    int secondaryShadowSamples = 1;
    float shadowSoftness = 0;
    int reflectionSamples = 8;
    int refractionSamples = 1;
    int reflectionDepth = 4;
    int refractionDepth = 8;
};

struct pfTri
{
    pfVec a;
    pfVec b;
    pfVec c;
    pfVec centroid;
    pfVec aN;
    pfVec bN;
    pfVec cN;
    int ID;
};

struct pfHit
{
    pfVec shadingNormal;
    pfVec geoNormal;
    pfVec barycoords;
    pfVec hitpos;
    int primIdx = -1; //ensure invalid index for default intersect
};

struct pfRay
{
    pfVec origin;
    pfVec direction;
    float t = INF;
    pfHit hit;
    bool isPrimary = false;
    bool terminated = false;
};

struct pfMatrix
{
    //each one is a column of the matrix
    pfVec v1;
    pfVec v2;
    pfVec v3;

    __host__ __device__ pfVec operator[](const int& index) const
    {
        if (index == 0) return v1;
        if (index == 1) return v2;
        if (index == 2) return v3;
    }

    __host__ __device__ pfVec operator*(const pfVec& a) const //matrix by vector
    {
        return
        {
            v1.x * a.x + v2.x * a.y + v3.x * a.z, //x
            v1.y * a.x + v2.y * a.y + v3.y * a.z, //y
            v1.z * a.x + v2.z * a.y + v3.z * a.z  //z
        };
    }

    __host__ __device__ const pfMatrix& operator*(pfMatrix a) const //matrix by matrix
    {
        pfMatrix out;

        out.v1.x = v1.x * a.v1.x + v2.x * a.v1.y + v3.x * a.v1.z;
        out.v1.y = v1.y * a.v1.x + v2.y * a.v1.y + v3.y * a.v1.z;
        out.v1.z = v1.z * a.v1.x + v2.z * a.v1.y + v3.z * a.v1.z;

        out.v2.x = v1.x * a.v2.x + v2.x * a.v2.y + v3.x * a.v2.z;
        out.v2.y = v1.y * a.v2.x + v2.y * a.v2.y + v3.y * a.v2.z;
        out.v2.z = v1.z * a.v2.x + v2.z * a.v2.y + v3.z * a.v2.z;

        out.v3.x = v1.x * a.v3.x + v2.x * a.v3.y + v3.x * a.v3.z;
        out.v3.y = v1.y * a.v3.x + v2.y * a.v3.y + v3.y * a.v3.z;
        out.v3.z = v1.z * a.v3.x + v2.z * a.v3.y + v3.z * a.v3.z;

        return out;
    }
};

struct pfCamera
{
    pfVec origin;
    pfMatrix matrix;
};

__host__ __device__ float dot(const pfVec& a, const pfVec& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ pfVec cross(const pfVec& a, const pfVec& b)
{
    return
    {
        a.y * b.z - a.z * b.y, //x
        a.z * b.x - a.x * b.z, //y
        a.x * b.y - a.y * b.x  //z
    };
}

__host__ __device__ float getNorm(pfVec& a)
{
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ void normalize(pfVec &a)
{
    a = a / getNorm(a);
}

__host__ __device__ void transpose(pfMatrix &a)
{
    pfMatrix out;
    out.v1 = { a.v1.x,a.v2.x ,a.v3.x };
    out.v2 = { a.v1.y,a.v2.y ,a.v3.y };
    out.v3 = { a.v1.z,a.v2.z ,a.v3.z };

    a = out;
}

__host__ __device__ float remap (const float &pInput, const float &pOldMin, const float &pOldMax, const float &pNewMin, const float &pNewMax)
{
    return ((pInput - pOldMin) / (pOldMax - pOldMin)) * (pNewMax - pNewMin) + pNewMin;
}

__host__ __device__ pfVec getTriNormal(const pfTri& pTri)
{
    pfVec n;
    n = cross(pTri.c - pTri.b, pTri.a - pTri.b);
    normalize(n);

    return n;
}

__host__ __device__ pfVec baryCoords(const pfTri &pTri, const pfVec &pPoint)
{
    //https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    pfTri t = pTri;
    pfVec p = pPoint;

    pfVec v0 = t.b - t.a;
    pfVec v1 = t.c - t.a;
    pfVec v2 = p - t.a;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    float denom = d00 * d11 - d01 * d01;

    pfVec baryCoords;

    baryCoords.y = (d11 * d20 - d01 * d21) / denom;
    baryCoords.z = (d00 * d21 - d01 * d20) / denom;
    baryCoords.x = 1 - baryCoords.y - baryCoords.z;

    return baryCoords;
}

__host__ __device__ pfVec remapNormal(const pfTri& pTri, const pfVec& pBaryCoords)
{

    pfVec out;
    out.x = pTri.aN[0] * pBaryCoords[0] + pTri.bN[0] * pBaryCoords[1] + pTri.cN[0] * pBaryCoords[2];
    out.y = pTri.aN[1] * pBaryCoords[0] + pTri.bN[1] * pBaryCoords[1] + pTri.cN[1] * pBaryCoords[2];
    out.z = pTri.aN[2] * pBaryCoords[0] + pTri.bN[2] * pBaryCoords[1] + pTri.cN[2] * pBaryCoords[2];

    normalize(out);
    return out;
}

__host__ __device__ pfVec remapPosition(const pfTri& pTri, const pfVec& pBaryCoords)
{

    pfVec out;
    out.x = pTri.a[0] * pBaryCoords[0] + pTri.b[0] * pBaryCoords[1] + pTri.c[0] * pBaryCoords[2];
    out.y = pTri.a[1] * pBaryCoords[0] + pTri.b[1] * pBaryCoords[1] + pTri.c[1] * pBaryCoords[2];
    out.z = pTri.a[2] * pBaryCoords[0] + pTri.b[2] * pBaryCoords[1] + pTri.c[2] * pBaryCoords[2];

    return out;
}

__device__ void invert(pfVec& a)
{
    a = a * -1;
}


__device__ pfVec snellsLaw(const pfVec& inDir, const pfVec& pN, const float inIOR, const float refIOR)
{
    float ThetaI = acosf(dot(pN, inDir * -1)); //incident angle

    float ThetaR = asinf((inIOR * sinf(ThetaI)) / refIOR); //refraction angle

    pfVec refX = cross(cross(pN, inDir), pN); normalize(refX);
    pfVec refY = pN * -1; normalize(refY);

    //Refracted direction
    pfVec out = refY + refX * tanf(ThetaR); normalize(out);

    //Check for total internal reflection
    //https://en.wikipedia.org/wiki/Total_internal_reflection
    if (refIOR <= inIOR)
    {
        float criticalAngle = asinf(refIOR / inIOR);
        if (ThetaI >= criticalAngle)
        {
            //reflect direction around normal                   
            return out = inDir - pN * dot(inDir, pN) * 2;
        }
    }

    return out;
}


///*
__device__ float sampleNoise(const int& pSample)
{
    //https://github.com/covexp/cuda-noise/tree/master
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * (threadIdx.x * gridDim.x);
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * (threadIdx.y * gridDim.y);

    int seed = (pSample * gridDim.y * gridDim.x);

    return cudaNoise::discreteNoise({fx,fy,0}, 1, seed);
}


__device__ pfVec randomHemisphereVector(const pfVec& pN, const float& rand1, const float& rand2)
{
    // Convert to spherical coordinates
    float r = sqrtf(1.0f - rand1 * rand1);
    float phi = 2.0f * PI * rand2;

    pfVec randomVec;
    randomVec.x = r * cosf(phi);
    randomVec.y = r * sinf(phi);
    randomVec.z = rand1;

    // Flip the vector if it's not in the same hemisphere as the normal
    return (dot(randomVec, pN) > 0.0f) ? randomVec : randomVec * -1.0f;
}
//*/

__device__ __forceinline__ uint wangHash(uint& seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ __forceinline__ float rand01(uint& seed)
{
    return wangHash(seed) * (1.0f / 4294967296.0f);
}

__device__ __forceinline__ pfVec cosineHemisphereSample(float u1, float u2)
{
    float r = sqrtf(u1);
    float theta = 2.0f * PI * u2;

    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);

    return pfVec{ x, y, z };
}

__device__ __forceinline__ void buildOrthonormalBasis(
    const pfVec& n,
    pfVec& t,
    pfVec& b
)
{
    if (fabs(n.z) < 0.999f)
    {
        t = cross(pfVec{ 0, 0, 1 }, n); normalize(t);
    }
    else
    { 
        t = cross(pfVec{ 0, 1, 0 }, n); normalize(t);
    }

    b = cross(n, t);
}

__device__ pfVec randomCosineHemisphereVector(
    const pfVec& normal,
    uint pixelIdx,
    uint bounce,
    uint pass
)
{
    uint seed = pixelIdx * 9781u + bounce * 6271u + pass * 15731u;

    float u1 = rand01(seed);
    float u2 = rand01(seed);

    pfVec local = cosineHemisphereSample(u1, u2);

    pfVec t, b;
    buildOrthonormalBasis(normal, t, b);

    pfVec out = t * local.x + b * local.y + normal * local.z;

	normalize(out);

    return out;
}








__device__ float orenNayarBRDF(const pfVec& pN, const pfVec& pToLight, const pfVec& pToView, float pRoughness)
{
    //https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models

    float NdotLight = fmaxf(dot(pN, pToLight), 0);
    float NdotView = fmaxf(dot(pN, pToView), 0);

    //angles 
    float thetaI = acosf(NdotLight); //incident (to light)
    float thetaO = acosf(NdotView); //outgoing (to view)

    float alpha = fmaxf(thetaI, thetaO);
    float beta = fminf(thetaI, thetaO);

    //coefficients
    float sigmaSqrd = pRoughness * pRoughness;
    float A = 1.0f - (sigmaSqrd / (2.0f * (sigmaSqrd + 0.33f)));
    float B = 0.45f * sigmaSqrd / (sigmaSqrd + 0.09f);
   
    float phiDif = atan2f(pToLight.y, pToLight.x) - atan2(pToView.y, pToView.x); //get azimuth angle difference

    float out = (A + B * fmaxf(0.0f, cosf(phiDif)) * sinf(alpha) * tanf(beta)) * NdotLight;

    return (out < 0) ? 0 : out;
}



__device__ float GGXBRDF(const pfVec& pN, const pfVec& pToLight, const pfVec& pToView, float pRoughness,float pIOR)
{
    //https://graphicscompendium.com/theory/08-cook-torrance-ggx
    pfVec half = pToLight + pToView; normalize(half);

    //Normal distribution function for GGX
    float alpha = pRoughness*pRoughness;
    float alphaSqrd = alpha * alpha;

    auto D = [&]() ->float //GGXdistribution
        {
            float NdotHalf = dot(pN, half);
  
            float NdotHalfSqrd = NdotHalf * NdotHalf;

            float X = (NdotHalf > 0) ? 1 : 0;

            float denom = PI * powf(NdotHalfSqrd * (alphaSqrd - 1) + 1, 2);

            float out = (X * alphaSqrd) / (denom + 1e-6f);

            out = (out > 1) ? 1 : out;

            return out;
        };


    auto G = [&](pfVec x) ->float //Geometric shadowing function
        {
            float xdotHalf = fmaxf(dot(x, half), 1e-6f);
            float xdotN = fmaxf(dot(x, pN), 1e-6f);
            float NdotHalf = fmaxf(dot(half, pN), 1e-6f);

            float X = (xdotHalf / xdotN > 0) ? 1 : 0;
            
            float NdotxSqrd = powf(xdotN, 2);

            float tanSqrd = (1 - NdotxSqrd) / NdotxSqrd;

            float denom = 1 + sqrtf(1 + alphaSqrd * tanSqrd);

            return X * (2 / denom);
        };


    auto F = [&]() ->float //Fresnel (Schlick)
        {
            float F0 = powf(pIOR - 1, 2) / powf(pIOR + 1, 2);
            
            float VdotHalf = fmaxf(dot(pToView, half), 1e-6f);

            return F0 + (1 - F0) * powf(1 - VdotHalf, 5);
        };

    return D() * G(pToLight) * G(pToView) * F();
}

__device__ pfVec sampleBRDF(const pfVec& pLightPos, const pfVec& pCamPos, const pfVec& pHitPos, const pfVec& pSurfaceNormal, const pfMaterial& pMat)
{
    pfVec toLight = pLightPos - pHitPos; normalize(toLight);
    pfVec toView = pCamPos - pHitPos; normalize(toView);


    float d = orenNayarBRDF(pSurfaceNormal, toLight, toView, pMat.diffuseRoughness); //diffuse component
    float s = GGXBRDF(pSurfaceNormal, toLight, toView, fmaxf(pMat.roughness, 0.01), pMat.IOR); //specular component

    auto F = [&]() ->float //Fresnel (Schlick)
        {
            pfVec half = toLight + toView; normalize(half);
            float F0 = powf(pMat.IOR - 1, 2) / powf(pMat.IOR + 1, 2);

            float VdotHalf = fmaxf(dot(toView, half), 1e-6f);

            return F0 + (1 - F0) * powf(1 - VdotHalf, 5);
        };

    float f = F();

    pfVec outColor = pMat.color * d * (1 - f) + pMat.specularColor * s * f;

    return outColor;
}

