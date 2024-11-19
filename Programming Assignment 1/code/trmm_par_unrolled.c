void trmm_par(int N, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c) 
{
    int i, j, k;

    #pragma omp parallel for schedule(guided) private(i, j, k)
    for (j = 0; j < N; j++) {
        for (i = 0; i <= j; i++) {
            double temp = 0.0;

            // Process remaining elements before unrolling
            int remainder = (j - i + 1) % 4;
            for (k = i; k < i + remainder; k++) {
                temp += (double)a[i * N + k] * b[k * N + j];
            }

            // Unrolling by 4 for higher performance
            #pragma omp simd reduction(+:temp)
            for (; k <= j; k += 4) {
                temp += (double)a[i * N + k] * b[k * N + j];
                temp += (double)a[i * N + (k + 1)] * b[(k + 1) * N + j];
                temp += (double)a[i * N + (k + 2)] * b[(k + 2) * N + j];
                temp += (double)a[i * N + (k + 3)] * b[(k + 3) * N + j];
            }

            // Final atomic update
            #pragma omp atomic
            c[i * N + j] += (float)temp;
        }
    }
}
