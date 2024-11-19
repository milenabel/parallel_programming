void vec2_ref(int n, float *__restrict__ A)
{
 int i, j;
  for (j=0; j<n; j++)
   for(i=1; i<n; i++)
//  A[i][j] = A[i-1][j]+1;
    A[i*n+j] = A[(i-1)*n+j]+1;
}
