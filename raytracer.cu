#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <algorithm>

constexpr int IMAGE_WIDTH = 800;
constexpr int IMAGE_HEIGHT = 600;
constexpr int BLOCK_DIM = 16;
constexpr float EPSILON = 0.001f;
constexpr int MAX_SPHERES = 10;

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    float3 toFloat3() const { return make_float3(x, y, z); }
    static Vec3 fromFloat3(const float3& v) { return Vec3(v.x, v.y, v.z); }
};

struct Ray { 
    float3 origin, direction; 
};

struct Sphere { 
    float3 center; 
    float radius; 
    float3 color; 
};

struct Light { 
    float3 position, color; 
};

struct Camera { 
    float3 position; 
    float fov; 
};

struct HitRecord { 
    float t; 
    float3 point, normal; 
    bool hit; 
    int sphereIndex; 
};

struct SceneData {
    Sphere spheres[MAX_SPHERES];
    int numSpheres;
    Light light;
    Camera camera;
};

__device__ float3 operator+(const float3& a, const float3& b) { 
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); 
}

__device__ float3 operator-(const float3& a, const float3& b) { 
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); 
}

__device__ float3 operator*(const float3& a, const float3& b) { 
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); 
}

__device__ float3 operator*(const float3& a, float t) { 
    return make_float3(a.x * t, a.y * t, a.z * t); 
}

__device__ float dot(const float3& a, const float3& b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

__device__ float length(const float3& v) { 
    return sqrtf(dot(v, v)); 
}

__device__ float3 normalize(const float3& v) { 
    return v * (1.0f / length(v)); 
}

__device__ float min_no_branch(float a, float b) { 
    return a + ((b - a) * (b < a)); 
}

__device__ float max_no_branch(float a, float b) { 
    return a + ((b - a) * (a < b)); 
}

__device__ float clamp_no_branch(float v, float min_v, float max_v) { 
    return min_no_branch(max_no_branch(v, min_v), max_v); 
}

__device__ HitRecord intersectRaySphere(const Ray& ray, const Sphere& sphere, int index) {
    HitRecord rec;
    rec.hit = false;
    rec.sphereIndex = -1;

    float3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0f * a * c;
    float sqrtDisc = sqrtf(max_no_branch(0.0f, discriminant));
    float t1 = (-b - sqrtDisc) / (2.0f * a);
    float t2 = (-b + sqrtDisc) / (2.0f * a);
    float t_near = (t1 > EPSILON) ? t1 : t2;
    float t_valid = (t_near > EPSILON) ? t_near : 0.0f;

    rec.hit = (discriminant >= 0.0f) && (t_valid > EPSILON);
    rec.t = t_valid;
    rec.point = ray.origin + ray.direction * t_valid;
    rec.normal = (rec.point - sphere.center) * (1.0f / sphere.radius);
    rec.sphereIndex = index;

    return rec;
}

__device__ float3 calculateDiffuseLight(const float3& hitPoint, const float3& normal, const float3& sphereColor, const Light& light) {
    float3 lightDir = normalize(light.position - hitPoint);
    float diffuseFactor = max_no_branch(0.0f, dot(normal, lightDir));
    return sphereColor * light.color * diffuseFactor;
}

__global__ void raytraceKernel(uchar4* outputBuffer, int width, int height, SceneData scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= width) || (y >= height)) return;

    float ndcX = (2.0f * x / width - 1.0f) * ((float)width / height);
    float ndcY = 1.0f - 2.0f * y / height;

    Ray ray;
    ray.origin = scene.camera.position;
    ray.direction = normalize(make_float3(ndcX, ndcY, -1.0f));

    HitRecord closestHit;
    closestHit.t = 1e30f;
    closestHit.hit = false;
    closestHit.sphereIndex = -1;

    for (int i = 0; i < scene.numSpheres; ++i) {
        HitRecord rec = intersectRaySphere(ray, scene.spheres[i], i);
        bool isCloser = rec.hit && (rec.t < closestHit.t);
        
        closestHit.t = isCloser ? rec.t : closestHit.t;
        closestHit.point = isCloser ? rec.point : closestHit.point;
        closestHit.normal = isCloser ? rec.normal : closestHit.normal;
        closestHit.hit = closestHit.hit || isCloser;
        closestHit.sphereIndex = isCloser ? rec.sphereIndex : closestHit.sphereIndex;
    }

    float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
    pixelColor = closestHit.hit ? calculateDiffuseLight(
        closestHit.point,
        closestHit.normal,
        scene.spheres[closestHit.sphereIndex].color,
        scene.light) : pixelColor;

    pixelColor.x = clamp_no_branch(pixelColor.x, 0.0f, 1.0f);
    pixelColor.y = clamp_no_branch(pixelColor.y, 0.0f, 1.0f);
    pixelColor.z = clamp_no_branch(pixelColor.z, 0.0f, 1.0f);

    outputBuffer[y * width + x] = make_uchar4(
        (unsigned char)(pixelColor.x * 255.0f),
        (unsigned char)(pixelColor.y * 255.0f),
        (unsigned char)(pixelColor.z * 255.0f),
        255
    );
}

void checkCudaError(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d %s\n", cudaGetErrorString(result), file, line, func);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCudaError(call, #call, __FILE__, __LINE__)

void saveToPPM(const std::string& filename, uchar4* pixels, int width, int height) {
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uchar4 p = pixels[y * width + x];
            file << (int)p.x << " " << (int)p.y << " " << (int)p.z << " ";
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Saved to " << filename << std::endl;
}

int main() {
    std::cout << "Ray Tracer Started" << std::endl;

    uchar4* h_output = new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT];
    uchar4* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uchar4)));

    SceneData scene;
    scene.numSpheres = 2;
    scene.spheres[0] = { make_float3(0, 0, -2), 1.0f, make_float3(0.8f, 0.3f, 0.2f) };
    scene.spheres[1] = { make_float3(-1.5f, 0.5f, -3), 0.5f, make_float3(0.2f, 0.8f, 0.3f) };
    scene.light = { make_float3(5, 5, 5), make_float3(1, 1, 1) };
    scene.camera = { make_float3(0, 0, 2), 60.0f };

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_HEIGHT + block.y - 1) / block.y);

    raytraceKernel<<<grid, block>>>(d_output, IMAGE_WIDTH, IMAGE_HEIGHT, scene);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost));

    saveToPPM("output.ppm", h_output, IMAGE_WIDTH, IMAGE_HEIGHT);

    delete[] h_output;
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Ray Tracer Finished" << std::endl;

    return 0;
}