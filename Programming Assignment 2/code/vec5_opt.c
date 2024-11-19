#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

void vec5_opt(int N, float *__restrict__ A, float *__restrict__ x) {
    const int B = 128; // Block size for better cache utilization
    int i, j, jb;

    // Initialize the result array with zeros
    #pragma omp parallel for schedule(static)
    for (i = 0; i < N; i++) {
        x[i] = 0.0f;
    }

    // Parallelized and vectorized computation
    #pragma omp parallel for schedule(static) private(j, jb) reduction(+:x[:N])
    for (jb = 0; jb < N; jb += B) {
        int block_end = (jb + B > N) ? N : jb + B;

        for (j = jb; j < block_end; j++) {
            #pragma omp simd aligned(A : 32)
            for (i = 0; i < N; i++) {
                x[i] += A[j * N + i] * A[j * N + i];
            }
        }
    }
}
