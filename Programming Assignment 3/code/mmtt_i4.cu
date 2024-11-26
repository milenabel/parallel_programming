#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001

void checkCUDAError(const char *msg);

const int DSIZE = 1024;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// Matrix multiply kernel: C = A^T * B^T with 4-way unrolling along i
__global__ void mmtt_i4(const float *A, const float *B, float *C, int ds) {
    int row_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_base < ds) {
        for (int col = 0; col < ds; col++) {
            float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;

            for (int k = 0; k < ds; k++) {
                value0 += A[k * ds + (row_base + 0)] * B[col * ds + k];
                if (row_base + 1 < ds) value1 += A[k * ds + (row_base + 1)] * B[col * ds + k];
                if (row_base + 2 < ds) value2 += A[k * ds + (row_base + 2)] * B[col * ds + k];
                if (row_base + 3 < ds) value3 += A[k * ds + (row_base + 3)] * B[col * ds + k];
            }

            C[(row_base + 0) * ds + col] = value0;
            if (row_base + 1 < ds) C[(row_base + 1) * ds + col] = value1;
            if (row_base + 2 < ds) C[(row_base + 2) * ds + col] = value2;
            if (row_base + 3 < ds) C[(row_base + 3) * ds + col] = value3;
        }
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
    int i, j, k;

    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    h_Cref = new float[DSIZE * DSIZE];

    for (i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
        h_C[i] = 0;
        h_Cref[i] = 0;
    }

    for (i = 0; i < DSIZE; i++) {
        for (k = 0; k < DSIZE; k++) {
            for (j = 0; j < DSIZE; j++) {
            //  h_Cref[i][j] += h_A[k][i]*h_B[j][k];
                h_Cref[i * DSIZE + j] += h_A[k * DSIZE + i] * h_B[j * DSIZE + k];
            }
        }
    }

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    checkCUDAError("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy H2D transfer failure");

    dim3 block(256, 1);
    dim3 grid((DSIZE + block.x - 1) / block.x, 1);

    printf("Matrix size: %d\n", DSIZE);

    for (int trial = 0; trial < 3; trial++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // Launch kernel
        mmtt_i4<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
        checkCUDAError("GPU kernel launch failure");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaDeviceSynchronize();
        // Copy results back to host
        cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy D2H");

        for (int i = 0; i < DSIZE * DSIZE; i++) {
            if (fabs((h_C[i] - h_Cref[i]) / h_Cref[i]) > threshold) {
                printf("Mismatch at %d: GPU %f CPU %f\n", i, h_C[i], h_Cref[i]);
                break;
            }
        }

        printf("Trial %d: Elapsed Time: %f ms, GFLOPS: %.2f\n", trial + 1, elapsedTime, 2.0e-6 * DSIZE * DSIZE * DSIZE / elapsedTime);
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
