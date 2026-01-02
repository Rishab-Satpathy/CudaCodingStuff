#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <limits>

using namespace std;
const float INF = std::numeric_limits<float>::infinity();

struct Vec3 {
    float x, y, z;

    Vec3 operator+(const Vec3& b) const {
        return {x + b.x, y + b.y, z + b.z};
    }

    Vec3 operator-(const Vec3& b) const {
        return {x - b.x, y - b.y, z - b.z};
    }

    Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;   

};
__device__ Sphere d_spheres[3] = {
    {{0,0,-3}, 1.0f, {255,255,0}},
    {{0.8,0.5,-3.5}, 3.0f, {255,225,0}},
    {{-0.8,-0.5,-4}, 2.0f, {215,255,3}}
};

__device__ float dot(const Vec3& a, const Vec3& b) 
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ Vec3 normalize(const Vec3& v)
{
    float len = sqrt(dot(v, v));
    return v * (1.0f / len);
}   

__device__ Vec3 ray_trace(int x, int y, int width, int height);
__device__ Vec3 color(const Sphere*, bool, Vec3, Vec3, float);

__global__ void render_kernel(Vec3* fb, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    fb[y * width + x] = ray_trace(x, y, width, height);
}


__device__ bool hitSphere(const Sphere& s, Vec3 rayOrig, Vec3 rayDir, float& t) 
{
    Vec3 oc = rayOrig - s.center;

    float a = dot(rayDir, rayDir);
    float b = 2.0f * dot(oc, rayDir);
    float c = dot(oc, oc) - s.radius * s.radius;

    float disc = b*b - 4*a*c;
    if (disc < 0) return false;

    t = (-b - sqrt(disc)) / (2*a);
    return t > 0;

}

__device__ Vec3 ray_trace(int x, int y,int height, int width)
{
    Vec3 cameraPos = {0, 0, 0};
    float viewportHeight = 2.0f;
    float viewportWidth = 2.0f;
    float focalLength = 1.0f;

     Vec3 rayDir = 
     {
        (x / (float)width  - 0.5f) * viewportWidth,
        (y / (float)height - 0.5f) * viewportHeight,
        -focalLength
    };

    rayDir = normalize(rayDir);

    float t; 
    
    float closest = INF;
    const Sphere* ClosestSphere = nullptr;

            for(int i =0;i<3;i++)
            {
                if (hitSphere(d_spheres[i], cameraPos, rayDir, t) && t < closest) {
                closest = t;
                ClosestSphere = &d_spheres[i];
            }
            }

            if(!ClosestSphere)
            {
                return {50, 80, 200}; //background
            }
    
            Vec3 hitPoint = cameraPos + rayDir * closest;
            Vec3 normal = normalize(hitPoint - ClosestSphere->center);

            //Point light stuff
            Vec3 lightPos = {5, 5, 0};
            Vec3 toLight = lightPos - hitPoint;
            float distance = sqrtf(dot(toLight, toLight));
            Vec3 lightDir = normalize(toLight);

            bool inShadow = false;
            Vec3 shadowOrigin = hitPoint + normal * 0.001f;

            for (int i =0;i<3;i++) 
            {
                float shadowT;
                if (&d_spheres[i] != ClosestSphere && hitSphere(d_spheres[i], shadowOrigin, lightDir, shadowT) && shadowT < distance)
                {
                inShadow = true;
                break;
                }
            }

    return color(ClosestSphere,inShadow,normal,lightDir,distance);

}

__device__ Vec3 color(Sphere *ClosestSphere,bool inShadow, Vec3 normal, Vec3 lightDir, float distance)
    {
                float intensity = fmaxf(dot(normal, lightDir), 0.0f);
                float attenuation = 1.0f / (1.0f + distance * distance);

                intensity *= attenuation;

            if (inShadow) 
            {
                intensity *= 0.2f; // ambient only
            }

                int r = int(ClosestSphere->color.x * intensity);
                int g = int(ClosestSphere->color.y * intensity);
                int b = int(ClosestSphere->color.z * intensity);

                Vec3 colors = {r,g,b};
                return (colors); //final sphere + lighting
    }
    
int main()
{
    int width = 1920;
    int height = 1080;
    int N = 1;

    std::ofstream img("output.ppm");
    img<<"P3\n"<<width<< " " << height << "\n255\n";

    Vec3* d_fb;
    Vec3* h_fb = new Vec3[width * height];

    cudaMalloc(&d_fb, sizeof(Vec3)*width*height);

    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);

    render_kernel<<<grid, block>>>(d_fb, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fb, d_fb, sizeof(Vec3)*width*height, cudaMemcpyDeviceToHost);

    for (int y = 0; y < height; y++) 
    {
    for (int x = 0; x < width; x++) 
    {
        Vec3 c = h_fb[y*width + x];
        img << (int)c.x << " " << (int)c.y << " " << (int)c.z << " ";
    }
    img << "\n";
    }

    return 0;
}