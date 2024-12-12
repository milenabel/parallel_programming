void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

// Initially identical to atb_seq; change this code to improve performance
  for (i = 0; i < Ni; i++)
   for (k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}

