#include <iostream>

__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+= stride) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    int N = 1<<20;
    int blockSize = 256;

    int numBlocks = (N + blockSize - 1) / blockSize;

    // std::cout << numBlocks << std::endl;
    // std::cout << N << std::endl;
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    std::cout << y[0] << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}