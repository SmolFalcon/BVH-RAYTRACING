#pragma once
//https://github.com/nothings/stb
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <ppl.h>
#include <vector>

class pfImg
{
public:

    int height;
    int width;
    pfVec* data;

    pfImg(int ph,int pw) //initialize image
    {
        height = ph;
        width = pw;
        data = new pfVec[height * width];
    }

    ~pfImg() 
    {
        delete[] data;
    }

    pfVec getXY(const int& pX, const int& pY) //Get value from X,Y coordinates, top left is 0,0
    {
        return data[pY + pX * height];
    }

    void setXY(const int& pX, const int& pY, const pfVec& pV) //Get value from X,Y coordinates, top left is 0,0
    {
        data[pY + pX * height] = pV;
    }

    __device__ static void CUDA_setXY(pfVec* pData, int pHeight, const int& pX, const int& pY, const pfVec& pV) //CUDA version
    {
        pData[pY + pX * pHeight] = pV;
    }

    __device__ static pfVec CUDA_getXY(pfVec* pData, int pHeight, const int& pX, const int& pY) //CUDA version
    {
        return pData[pY + pX * pHeight];
    }

    void normalizeImage()
    {
        float lmax = -INF;
        float lmin = INF;
        
        for (int i = 0; i < height*width; i++)
        {
            lmax = (data[i].x > lmax && data[i].x < INF) ? data[i].x : lmax;
            lmax = (data[i].y > lmax && data[i].y < INF) ? data[i].y : lmax;
            lmax = (data[i].z > lmax && data[i].z < INF) ? data[i].z : lmax;

            lmin = (data[i].x < lmin) ? data[i].x : lmin;
            lmin = (data[i].y < lmin) ? data[i].y : lmin;
            lmin = (data[i].z < lmin) ? data[i].z : lmin;
        }

        for (int i = 0; i < height * width; i++)
        {
            data[i].x = remap(data[i].x, lmin, lmax, 0, 255);
            data[i].y = remap(data[i].y, lmin, lmax, 0, 255);
            data[i].z = remap(data[i].z, lmin, lmax, 0, 255);
        }
    }

    void storeImage(const char* pFile)
    {
        int lChannels = 3;
        int lRows = height;
        int lCols = width;

        std::vector<unsigned char> image(lRows * lCols * lChannels, 0);

        concurrency::parallel_for(0, lRows, [&](int i)
        //for (int i = 0; i < lRows; i++)
        {
            for (int j = 0; j < lCols; j++)
            {
                pfVec lPixel = data[i + j * lRows];

                int lOutIdx = (j + i * lCols) * 3; //the out image index is flipped to transpose the image from rowsxcols to colsxrows

                //Store RGB color
                image[lOutIdx] = lPixel.x;
                image[lOutIdx + 1] = lPixel.y;
                image[lOutIdx + 2] = lPixel.z;
            }
        });

        // Save the image as PNG
        stbi_write_png(pFile, lCols, lRows, lChannels, image.data(), lCols * lChannels);
    }

    void loadImage(const char* pFile)
    {
        int lChannels = 3;
        int lRows = height;
        int lCols = width;

        unsigned char* img = stbi_load(pFile, &lCols, &lRows, &lChannels, 0);

        concurrency::parallel_for(0, lRows, [&](int i)
        //for (int i = 0; i < lRows; i++)
        {
            for (int j = 0; j < lCols; j++)
            {
                //flip indices to transpose image
                int lIndIdx = (j + i * lCols) * 3;

                data[i + j * lRows].x = img[lIndIdx];
                data[i + j * lRows].y = img[lIndIdx + 1];
                data[i + j * lRows].z = img[lIndIdx + 2];
            }
        });

        stbi_image_free(img);
    }
};

void getImageData(int& pRows, int& pColumns, const char* pFile = "input.png")
{
    stbi_info(pFile, &pColumns, &pRows, NULL);
}

void storeImage(float* pmatrix, int pRows, int pColumns, const char* pFile = "output.png")
{
    std::vector<unsigned char> image(pRows * pColumns, 0);

    for (int i = 0; i < pRows; i++)
    {
        for (int j = 0; j < pColumns; j++)
        {

            //flip indices to transpose image
            image[j + i * pColumns] = pmatrix[i + j * pRows];
        }
    }

    // Save the image as PNG
    stbi_write_png(pFile, pColumns, pRows, 1, image.data(), pColumns);
}

void loadImage(float* pmatrix, int pRows, int pColumns, const char* pFile = "input.png")
{
    int channels = 1;
    unsigned char* img = stbi_load(pFile, &pColumns, &pRows, &channels, 0);

    std::vector<unsigned char> image(img, img + (pRows * pColumns));

    for (int i = 0; i < pRows; i++)
    {
        for (int j = 0; j < pColumns; j++)
        {
            //flip indices to transpose image
            pmatrix[i + j * pRows] = image[j + i * pColumns];
        }
    }

    stbi_image_free(img);
}