#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001

void checkCUDAError(const char *msg);

const int DSIZE = 1024;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// Matrix multiply kernel: C = A^T * B^T with 4-way unrolling along i and j
__global__ void mmtt_i4j4(const float *A, const float *B, float *C, int ds) {
    int row_base = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_base < ds) {
        for (int col_base = 0; col_base < ds; col_base += 4) {
            float value00 = 0.0f, value01 = 0.0f, value02 = 0.0f, value03 = 0.0f;
            float value10 = 0.0f, value11 = 0.0f, value12 = 0.0f, value13 = 0.0f;
            float value20 = 0.0f, value21 = 0.0f, value22 = 0.0f, value23 = 0.0f;
            float value30 = 0.0f, value31 = 0.0f, value32 = 0.0f, value33 = 0.0f;

            for (int k = 0; k < ds; k++) {
                float a0 = A[k * ds + (row_base + 0)];
                float a1 = (row_base + 1 < ds) ? A[k * ds + (row_base + 1)] : 0.0f;
                float a2 = (row_base + 2 < ds) ? A[k * ds + (row_base + 2)] : 0.0f;
                float a3 = (row_base + 3 < ds) ? A[k * ds + (row_base + 3)] : 0.0f;

                value00 += a0 * B[(col_base + 0) * ds + k];
                value01 += a0 * B[(col_base + 1) * ds + k];
                value02 += a0 * B[(col_base + 2) * ds + k];
                value03 += a0 * B[(col_base + 3) * ds + k];

                if (row_base + 1 < ds) {
                    value10 += a1 * B[(col_base + 0) * ds + k];
                    value11 += a1 * B[(col_base + 1) * ds + k];
                    value12 += a1 * B[(col_base + 2) * ds + k];
                    value13 += a1 * B[(col_base + 3) * ds + k];
                }
                if (row_base + 2 < ds) {
                    value20 += a2 * B[(col_base + 0) * ds + k];
                    value21 += a2 * B[(col_base + 1) * ds + k];
                    value22 += a2 * B[(col_base + 2) * ds + k];
                    value23 += a2 * B[(col_base + 3) * ds + k];
                }
                if (row_base + 3 < ds) {
                    value30 += a3 * B[(col_base + 0) * ds + k];
                    value31 += a3 * B[(col_base + 1) * ds + k];
                    value32 += a3 * B[(col_base + 2) * ds + k];
                    value33 += a3 * B[(col_base + 3) * ds + k];
                }
            }

            C[(row_base + 0) * ds + (col_base + 0)] = value00;
            C[(row_base + 0) * ds + (col_base + 1)] = value01;
            C[(row_base + 0) * ds + (col_base + 2)] = value02;
            C[(row_base + 0) * ds + (col_base + 3)] = value03;

            if (row_base + 1 < ds) {
                C[(row_base + 1) * ds + (col_base + 0)] = value10;
                C[(row_base + 1) * ds + (col_base + 1)] = value11;
                C[(row_base + 1) * ds + (col_base + 2)] = value12;
                C[(row_base + 1) * ds + (col_base + 3)] = value13;
            }
            if (row_base + 2 < ds) {
                C[(row_base + 2) * ds + (col_base + 0)] = value20;
                C[(row_base + 2) * ds + (col_base + 1)] = value21;
                C[(row_base + 2) * ds + (col_base + 2)] = value22;
                C[(row_base + 2) * ds + (col_base + 3)] = value23;
            }
            if (row_base + 3 < ds) {
                C[(row_base + 3) * ds + (col_base + 0)] = value30;
                C[(row_base + 3) * ds + (col_base + 1)] = value31;
                C[(row_base + 3) * ds + (col_base + 2)] = value32;
                C[(row_base + 3) * ds + (col_base + 3)] = value33;
            }
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
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
    printf("Matrix size: %d\n", DSIZE);

    for (int trial = 0; trial < 3; trial++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // Launch kernel
        mmtt_i4j4<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
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

        printf("<BX=%d,BY=%d>: Trial %d: Elapsed Time: %f ms, GFLOPS: %.2f\n", block.x, block.y, trial + 1, elapsedTime, 2.0e-6 * DSIZE * DSIZE * DSIZE / elapsedTime);
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
