void trmm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
    int i, j, k;

    // Parallelize the outer loop with dynamic scheduling and chunk size 1
    #pragma omp parallel for private(j, k) schedule(dynamic, 1)
    for (i = 0; i < N; i++)
        for (j = i; j < N; j++)
            for (k = i; k <= j; k++)
                c[i * N + j] += a[i * N + k] * b[k * N + j];
}