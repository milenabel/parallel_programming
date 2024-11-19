void vec5_ref(int N, float *__restrict__ A, float *__restrict__ x)
{
int i, j; float sum;

 for (i=0; i<N; i++) 
 {
  sum = 0.0;
  for(j=0; j<N; j++)
//  sum += A[j][i]*A[j][i];
    sum += A[j*N+i]*A[j*N+i];
  x[i] = sum;
 }
}
