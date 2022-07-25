#include <iostream>
#include <X11/Xlib.h>
#include <unistd.h>
#include <math.h>

#define HEIGHT 10
#define WIDTH 30

__global__ void reset(char *screen) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    screen[y * WIDTH + x] = '.';
}


__device__ float distance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

__global__ void drawLine(char *screen, int x1, int y1, int x2, int y2) {
    int x3 = blockIdx.x * blockDim.x + threadIdx.x;
    int y3 = blockIdx.y * blockDim.y + threadIdx.y;

    if (static_cast<int>(distance(x1, y1, x3, y3)) + static_cast<int>(distance(x2, y2, x3, y3)) == static_cast<int>(distance(x1, y1, x2, y2))) {
        screen[y3 * WIDTH + x3] = '#';
    }
    
    // if distance of ac + distance of bc == distance of ab it crosses through line
    // distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
}

class Point2d {
    public:
        int x;
        int y;

        Point2d(int _x, int _y) {
            x = _x;
            y = _y;
        }
};

class Screen2d {
    public:
        Screen2d(int _width, int _height) {
            width = _width;
            height = _height;

            cudaMallocManaged(&screen, height * width * sizeof(int));
            numBlocks = ((WIDTH * HEIGHT) + blockSize - 1) / blockSize;
            drawBase();
        }

        void drawBase() {
            reset<<<numBlocks, blockSize>>>(screen);

            cudaDeviceSynchronize();

            std::cout << "drew" << std::endl;
        }

        void line(int x1, int y1, int x2, int y2) {
            drawLine<<<numBlocks, blockSize>>>(screen, x1, y1, x2, y2);

            cudaDeviceSynchronize();

            std::cout << "drew line" << std::endl;
        }

        void print() {
            for (int i = 0; i < HEIGHT; i++) {
                for (int k = 0; k < WIDTH; k++) {
                    putchar(screen[i * WIDTH + k]);
                    // screen[i * WIDTH + k] = '#';
                }
                putchar('\n');
            }
        }
    
    private:
        int width;
        int height;
        char *screen;

        int blockSize = 4;
        int numBlocks;
};

__global__ void populateVec(int n, float *a, float *b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;

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
    Screen2d screen(WIDTH, HEIGHT);
    screen.drawBase();
    screen.line(0, 0, 15, 5);
    screen.print();
    // char *screen = (char*)malloc(HEIGHT * WIDTH * sizeof(int));

    // for (int i = 0; i < HEIGHT; i++) {
    //     for (int k = 0; k < WIDTH; k++) {
    //         screen[i * WIDTH + k] = '.';
    //     }
    // }

    // Point2d *point1 = new Point2d(0, 0);
    // Point2d *point2 = new Point2d(WIDTH - 1, HEIGHT - 1);


    // screen[0 * WIDTH + 0] = '#';
    // screen[(HEIGHT - 1) * WIDTH + (WIDTH - 1)] = '#';

    // for (int i = 0; i < HEIGHT; i++) {
    //     for (int k = 0; k < WIDTH; k++) {
    //         putchar(screen[i * WIDTH + k]);
    //         // screen[i * WIDTH + k] = '#';
    //     }
    //     putchar('\n');
    // }

    // int nDevices;

    // cudaGetDeviceCount(&nDevices);

    // for (int k = 0; k < nDevices; k++) {
    //     cudaDeviceProp prop;
    //     cudaGetDeviceProperties(&prop, k);
    //     std::cout << prop.name << std::endl;
    // }

    // int n = 1<<20;
    // int blockSize = 512;
    // int numBlocks = (n + blockSize - 1) / blockSize;

    // float *a, *b, *c;

    // cudaMallocManaged(&a, n * sizeof(float));
    // cudaMallocManaged(&b, n * sizeof(float));
    // cudaMallocManaged(&c, n * sizeof(float));

    // populateVec<<<numBlocks, blockSize>>>(n, a, b);
    // vecAdd<<<numBlocks, blockSize>>>(n, a, b, c);

    // cudaDeviceSynchronize();

    // // for (int i = 0; i < 256; i++) {
    // //     std::cout << c[i] << std::endl;
    // // }

    // return 0;

    // vecAdd<<<1, 256>>>()
}