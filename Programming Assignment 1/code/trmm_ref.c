void trmm_ref(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) {
int i, j, k;

for (i = 0; i < N; i++)
 for (j = i; j < N; j++)
  for (k = i; k <= j; k++)
// c[i][j] = c[i][j] + a[i][k]*b[k][j];
   c[i*N+j]=c[i*N+j]+a[i*N+k]*b[k*N+j];
}
