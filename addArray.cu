#include <iostream>

__global__ void populateVec(int n, float *a, float *b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i+= stride) {
        a[i] = 2;
        b[i] = 5;
    }
}

__global__ void vecAdd(int n, float *a, float *b, float *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i+= stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // int nDevices;

    // cudaGetDeviceCount(&nDevices);

    // for (int k = 0; k < nDevices; k++) {
    //     cudaDeviceProp prop;
    //     cudaGetDeviceProperties(&prop, k);
    //     std::cout << prop.name << std::endl;
    // }

    int n = 1<<20;
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;

    float *a, *b, *c;

    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));

    populateVec<<<numBlocks, blockSize>>>(n, a, b);
    vecAdd<<<numBlocks, blockSize>>>(n, a, b, c);

    cudaDeviceSynchronize();

    // for (int i = 0; i < 256; i++) {
    //     std::cout << c[i] << std::endl;
    // }

    return 0;

    // vecAdd<<<1, 256>>>()
}