// jik Order with Guided Scheduling
void trmm_par(int N, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c) 
{
    int i, j, k;

    #pragma omp parallel for schedule(guided, 20) private(i, j, k)
    for (j = 0; j < N; j++) {
        for (i = 0; i <= j; i++) {
            for (k = i; k <= j; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}