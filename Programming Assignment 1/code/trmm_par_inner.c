void trmm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
    int i, j, k;

    #pragma omp parallel private(i, j)
    {
        for (i = 0; i < N; i++)
            for (j = i; j < N; j++)
                // Parallelize the inner loop directly with omp for
                #pragma omp for
                for (k = i; k <= j; k++)
                    c[i * N + j] += a[i * N + k] * b[k * N + j];
    }
}