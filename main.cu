#include <iostream>

__global__ void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(int *argc, char** argv[]) {
    int N = 1<<20;
    std::cout << N << std::endl;
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    cudaDeviceSynchronize();

    std::cout << y[0] << std::endl;

    return 0;
}