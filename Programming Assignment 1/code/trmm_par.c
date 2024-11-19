void trmm_par(int N, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) 
{
int i, j, k;

#pragma omp parallel private(i,j,k) 
 {
// Delete #pragma omp master
// and edit this code appropriately for the various parts of Question 2
#pragma omp master
  for (i = 0; i < N; i++)
   for (j = i; j < N; j++)
    for (k = i; k <= j; k++)
// c[i][j] = c[i][j] + a[i][k]*b[k][j];
     c[i*N+j]=c[i*N+j]+a[i*N+k]*b[k*N+j];
 }
}
