void vec2_opt(int n, float *__restrict__ A)
{
    int i, j;
    for (i=1; i<n; i++) {         // Outer loop over rows
        for (j=0; j<n; j++) {     // Inner loop over columns (can be vectorized)
            //  A[i][j] = A[i-1][j]+1;
            A[i*n+j] = A[(i-1)*n+j]+1;
        }
    }
}