#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001

void checkCUDAError(const char *msg);

const int DSIZE = 1024;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// Matrix multiply kernel using shared memory
__global__ void mmtt_sm(const float *A, const float *B, float *C, int ds) {
    __shared__ float Asub[16][16]; // Shared memory for A
    __shared__ float Bsub[16][16]; // Shared memory for B

    int tx = threadIdx.x;  // Local thread ID in x-dimension
    int ty = threadIdx.y;  // Local thread ID in y-dimension
    int row = blockIdx.y * blockDim.y + ty;  // Row index of C
    int col = blockIdx.x * blockDim.x + tx;  // Column index of C

    float value = 0.0f;  // Accumulator for the dot product

    // Iterate over tiles of A and B
    for (int t = 0; t < (ds + 15) / 16; t++) {
        // Load a tile of A and B into shared memory
        if (row < ds && t * 16 + tx < ds) {
            Asub[ty][tx] = A[(t * 16 + tx) * ds + row];
        } else {
            Asub[ty][tx] = 0.0f;
        }

        if (col < ds && t * 16 + ty < ds) {
            Bsub[ty][tx] = B[col * ds + (t * 16 + ty)];
        } else {
            Bsub[ty][tx] = 0.0f;
        }

        __syncthreads();  // Synchronize to ensure shared memory is fully loaded

        // Perform dot product for the current tile
        for (int k = 0; k < 16; k++) {
            value += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads();  // Synchronize before loading the next tile
    }

    // Write the result back to global memory
    if (row < ds && col < ds) {
        C[row * ds + col] = value;
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;

    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    h_Cref = new float[DSIZE * DSIZE];

    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
        h_C[i] = 0;
        h_Cref[i] = 0;
    }

    for (int i = 0; i < DSIZE; i++) {
        for (int k = 0; k < DSIZE; k++) {
            for (int j = 0; j < DSIZE; j++) {
                h_Cref[i * DSIZE + j] += h_A[k * DSIZE + i] * h_B[j * DSIZE + k];
            }
        }
    }

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    checkCUDAError("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy H2D transfer failure");

    // Block and grid dimensions
    dim3 block(16, 16);  
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    printf("Matrix size: %d\n", DSIZE);

    for (int trial = 0; trial < 3; trial++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        mmtt_sm<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
        checkCUDAError("GPU kernel launch failure");

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy D2H");

        for (int i = 0; i < DSIZE * DSIZE; i++) {
            if (fabs((h_C[i] - h_Cref[i]) / h_Cref[i]) > threshold) {
                printf("Mismatch at %d: GPU %f CPU %f\n", i, h_C[i], h_Cref[i]);
                break;
            }
        }

        printf("Trial %d: Elapsed Time: %f ms, GFLOPS: %.2f\n", trial + 1, elapsedTime, 2.0e-6 * DSIZE * DSIZE * DSIZE / elapsedTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_Cref;

    return 0;
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
