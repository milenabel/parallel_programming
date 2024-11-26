#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001

void checkCUDAError(const char *msg);

const int DSIZE = 1024;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// Matrix multiply kernel: C = A^T * B^T with 4-way unrolling along j
__global__ void mmtt_j4(const float *A, const float *B, float *C, int ds) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ds) {
        for (int col_base = 0; col_base < ds; col_base += 4) {
            float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;

            for (int k = 0; k < ds; k++) {
                float a_val = A[k * ds + row];
                value0 += a_val * B[(col_base + 0) * ds + k];
                value1 += a_val * B[(col_base + 1) * ds + k];
                value2 += a_val * B[(col_base + 2) * ds + k];
                value3 += a_val * B[(col_base + 3) * ds + k];
            }

            if (col_base + 0 < ds) C[row * ds + (col_base + 0)] = value0;
            if (col_base + 1 < ds) C[row * ds + (col_base + 1)] = value1;
            if (col_base + 2 < ds) C[row * ds + (col_base + 2)] = value2;
            if (col_base + 3 < ds) C[row * ds + (col_base + 3)] = value3;
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
        mmtt_j4<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
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
