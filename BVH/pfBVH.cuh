#include "pfUtils.cuh"
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stack>

//https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

class pfBVH
{
public:
    struct BVHNode
    {
        //Vectors to store bounding box limits
        pfVec aabbMin; 
        pfVec aabbMax;
        int leftChild = -1;

        int firstTriIdx, triCount;
    };

    pfTri* tri; //Geo array
    int* triIdx; //indices array, stores indices and will be reordered for efficient indexing in nodes while keeping the original geometry array unchanged
    int N; //triCount

    BVHNode* bvhNode;
    int rootNodeIdx = 0, nodesUsed = 1;

    //Initialize node array and triangle index array
    pfBVH(pfTri* pTri, int pN)
    {
        tri = pTri;
        N = pN;
        triIdx = new int[N];
        bvhNode = new BVHNode[N * 2 - 1];
    }

    ~pfBVH()//release memory
    {
        delete[]triIdx;
        delete[]bvhNode;
    }

    void BuildBVH()
    {     
        //Compute centroid of each triangle and populate triangle index array
        for (int i = 0; i < N; i++) 
        {
            tri[i].centroid = (tri[i].a + tri[i].b + tri[i].c) * 0.3333f;

            triIdx[i] = i;
        }

        // assign all triangles to root node
        BVHNode& root = bvhNode[rootNodeIdx];
        root.firstTriIdx = 0, root.triCount = N; //triangles are stored consecutively (start at 0 -> all triangles)

        //Initial bounding box
        UpdateNodeBounds(rootNodeIdx); 

        // subdivide recursively
        Subdivide(rootNodeIdx);
    }

    void IntersectBVH(pfRay & ray, const int nodeIdx)
    {
        BVHNode& node = bvhNode[nodeIdx];
        
        if (!IntersectAABB(ray, node.aabbMin, node.aabbMax))
        {
            return;
        }
        if (node.triCount > 0) //Node is leaf because contains triangles
        {
            for (int i = 0; i < node.triCount; i++)
            {
                IntersectTri(ray, tri[triIdx[node.firstTriIdx + i]], triIdx[node.firstTriIdx + i]);
            }
        }
        else
        {
            IntersectBVH(ray, node.leftChild);
            IntersectBVH(ray, node.leftChild + 1);
        }
    }


    __device__ static void CUDA_IntersectBVH(pfTri* pTriArr, int* pTriIdxArr, BVHNode* pBVHNodes, pfRay& ray) //CUDA version, research stack optimizations
    {
        const int MAX_STACK_SIZE = 64;  // Adjust based on BVH depth (64 seems enough)
        int stack[MAX_STACK_SIZE];
        int stackPtr = 0;

        stack[stackPtr++] = 0; //Push root node

        while (stackPtr > 0)
        {
            int nodeIdx = stack[--stackPtr]; //Pop node

            //AABB intersection test
            if (!IntersectAABB(ray, pBVHNodes[nodeIdx].aabbMin, pBVHNodes[nodeIdx].aabbMax))
                continue; //Skip this node if there's no intersection

            if (pBVHNodes[nodeIdx].triCount > 0) //Leaf node
            {
                for (int i = 0; i < pBVHNodes[nodeIdx].triCount; i++)
                {
                    IntersectTri(ray, pTriArr[pTriIdxArr[pBVHNodes[nodeIdx].firstTriIdx + i]], pTriIdxArr[pBVHNodes[nodeIdx].firstTriIdx + i]);
                }
            }
            else //Internal node, push children onto stack
            {
                //Push children in reverse order so the left child is processed first
                if (stackPtr < MAX_STACK_SIZE - 1)
                    stack[stackPtr++] = pBVHNodes[nodeIdx].leftChild + 1;  //Right child

                if (stackPtr < MAX_STACK_SIZE - 1)
                    stack[stackPtr++] = pBVHNodes[nodeIdx].leftChild;      //Left child
            }
        }
    }

    //Functions to cache structure to binary file
    void SaveBVH(const std::string& filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return;

        // Write triangle count and node count
        file.write(reinterpret_cast<char*>(&N), sizeof(int));
        file.write(reinterpret_cast<char*>(&nodesUsed), sizeof(int));

        // Write triangle array
        file.write(reinterpret_cast<char*>(tri), sizeof(pfTri) * N);

        // Write triangle indices
        file.write(reinterpret_cast<char*>(triIdx), sizeof(int) * N);

        // Write BVH nodes
        file.write(reinterpret_cast<char*>(bvhNode), sizeof(BVHNode) * nodesUsed);

        file.close();
    }
    void LoadBVH(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return;

        // Read triangle count and node count
        file.read(reinterpret_cast<char*>(&N), sizeof(int));
        file.read(reinterpret_cast<char*>(&nodesUsed), sizeof(int));

        // Read triangle array
        file.read(reinterpret_cast<char*>(tri), sizeof(pfTri) * N);

        // Read triangle indices
        file.read(reinterpret_cast<char*>(triIdx), sizeof(int) * N);

        // Read BVH nodes
        file.read(reinterpret_cast<char*>(bvhNode), sizeof(BVHNode) * nodesUsed);

        file.close();
    }
private:
    void UpdateNodeBounds(int nodeIdx)
    {
        BVHNode& node = bvhNode[nodeIdx];
        node.aabbMin; node.aabbMin.x = INF; node.aabbMin.y = INF; node.aabbMin.z = INF;
        node.aabbMax; node.aabbMax.x = -INF; node.aabbMax.y = -INF; node.aabbMax.z = -INF;

        auto lmin = [](pfVec a, pfVec b)->pfVec
            {
                pfVec out;
                out.x = (a.x < b.x) ? a.x : b.x;
                out.y = (a.y < b.y) ? a.y : b.y;
                out.z = (a.z < b.z) ? a.z : b.z;
                return out;
            };

        auto lmax = [](pfVec a, pfVec b)->pfVec
            {
                pfVec out;
                out.x = (a.x > b.x) ? a.x : b.x;
                out.y = (a.y > b.y) ? a.y : b.y;
                out.z = (a.z > b.z) ? a.z : b.z;
                return out;
            };

        for (int first = node.firstTriIdx, i = 0; i < node.triCount; i++)
        {
            int leafTriIdx = triIdx[first + i];
            pfTri& leafTri = tri[leafTriIdx];

            node.aabbMin = lmin(node.aabbMin, leafTri.a);
            node.aabbMin = lmin(node.aabbMin, leafTri.b);
            node.aabbMin = lmin(node.aabbMin, leafTri.c);
            node.aabbMax = lmax(node.aabbMax, leafTri.a);
            node.aabbMax = lmax(node.aabbMax, leafTri.b);
            node.aabbMax = lmax(node.aabbMax, leafTri.c);
        }
    }

    void Subdivide(int nodeIdx)
    {
        // terminate recursion if the node has 2 or less triangles inside
        BVHNode& node = bvhNode[nodeIdx];
        if (node.triCount <= 2) return;

        // determine split axis and position
        pfVec extent = node.aabbMax - node.aabbMin; //Corner to corner extent of the bounding box
        
        //determine the axis in which the bounding box is larger and split it in half there
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;
        float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;


        //in-place partition, reorder the triangle indices so the splits of the bounding box are consecutive
        int i = node.firstTriIdx;
        int j = i + node.triCount - 1;
        while (i <= j)
        {
            if (tri[triIdx[i]].centroid[axis] < splitPos)
            { 
                i++;
            }
            else //swap
            { 
                int lTemp = triIdx[i];
                triIdx[i] = triIdx[j];
                triIdx[j] = lTemp;
                j--;
            }
        }

        // abort split if one of the sides is empty
        int leftCount = i - node.firstTriIdx;
        if (leftCount == 0 || leftCount == node.triCount) return;

        // create child nodes
        int leftChildIdx = nodesUsed++;
        int rightChildIdx = nodesUsed++;
        bvhNode[leftChildIdx].firstTriIdx = node.firstTriIdx;
        bvhNode[leftChildIdx].triCount = leftCount;
        bvhNode[rightChildIdx].firstTriIdx = i;
        bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
        node.leftChild = leftChildIdx;
        node.triCount = 0;
        UpdateNodeBounds(leftChildIdx);
        UpdateNodeBounds(rightChildIdx);

        // recurse
        Subdivide(leftChildIdx);
        Subdivide(rightChildIdx);
    }


    __device__ __host__ static bool IntersectAABB(const pfRay & ray, const pfVec& bmin, const pfVec& bmax) //Get a simple intersection from the bounding box (Slab method)
    {
        auto lmin = [](float a, float b)->float
        {
            return (a < b) ? a : b;
        };

        auto lmax = [](float a, float b)->float
        {
            return (a > b) ? a : b;
        };
        
        float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
        float tmin = lmin(tx1, tx2), tmax = lmax(tx1, tx2);
        float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
        tmin = lmax(tmin, lmin(ty1, ty2)), tmax = lmin(tmax, lmax(ty1, ty2));
        float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
        tmin = lmax(tmin, lmin(tz1, tz2)), tmax = lmin(tmax, lmax(tz1, tz2));

        return tmax >= tmin && tmin < ray.t && tmax > 0;
    }


    public:
    __device__ __host__ static void IntersectTri(pfRay& pRay, pfTri& pTri, int idx) //Möller-Trumbore
    {
        float epsilon = 1e-6f;

        pfVec edge1 = pTri.b - pTri.a;
        pfVec edge2 = pTri.c - pTri.a;

        pfVec ray_cross_e2 = cross(pRay.direction, edge2);
        float det = dot(edge1, ray_cross_e2);

        if (det > -epsilon && det < epsilon) return; // ray is parallel to triangle

        float inv_det = 1.0 / det;

        pfVec s = pRay.origin - pTri.a;
        float u = inv_det * dot(s, ray_cross_e2);

        if ((u < 0.0f || u > 1.0f)) return;

        pfVec s_cross_e1 = cross(s, edge1);
        float v = inv_det * dot(pRay.direction, s_cross_e1);

        if (v < 0.0f || (u + v)>1.0f) return;

        float t = inv_det * dot(edge2, s_cross_e1);

        if (t > epsilon && t < pRay.t)
        {
            pRay.t = t; //Get the shortest distance
            pRay.hit.primIdx = idx;

            pRay.hit.barycoords = baryCoords(pTri, pRay.origin + pRay.direction * pRay.t);

            pRay.hit.hitpos = remapPosition(pTri, pRay.hit.barycoords); //more precise hitpos

            pRay.hit.geoNormal = getTriNormal(pTri);
            pRay.hit.shadingNormal = remapNormal(pTri, pRay.hit.barycoords);
        }   
    }
};



