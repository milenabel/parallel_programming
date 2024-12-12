// Use "gcc -O3 -fopenmp atb_main.c atb_par.c " or
// or use "clang -O3  -fopenmp atb_main.c atb_par.c" to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define NTrials (5)
#define threshold (0.0000001)

void atb_par(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk);

void atb_seq(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}


int main(int argc, char *argv[]){
  double tstart,telapsed;

  int i,j,k,nt,trial,nthreads;
  double mint_par,maxt_par, t_seq;

  double *A, *B, *C, *Cref;
  int Ni,Nj,Nk;

  printf("Enter Matrix dimensions Ni Nj Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  A = (double *) malloc(sizeof(double)*Nk*Ni);
  B = (double *) malloc(sizeof(double)*Nk*Nj);
  C = (double *) malloc(sizeof(double)*Ni*Nj);
  Cref = (double *) malloc(sizeof(double)*Ni*Nj);
  for (k=0; k<Nk; k++)
   for (i=0; i<Ni; i++)
    A[k*Ni+i] = k*Ni+i-1;
  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    B[k*Nj+j] = k*Nj+j+1;
  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++) {
    C[i*Nj+j] = 0;
    Cref[i*Nj+j] = 0;}

  for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) Cref[i*Nj+j] = 0;
  tstart = omp_get_wtime();
  atb_seq(A,B,Cref,Ni,Nj,Nk);
  t_seq = omp_get_wtime()-tstart;
  printf("Reference sequential code for ATB: Time = %.2f sec; Performance (in GFLOPS): %.2f\n",t_seq,2.0e-9*Ni*Nj*Nk/t_seq);

  nthreads = omp_get_max_threads();
  printf("Using %d Threads (from omp_get_max_threads)\n",nthreads);

  omp_set_num_threads(nthreads);
  mint_par = 1e9; maxt_par = 0;
  for (trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) C[i*Nj+j] = 0;
   tstart = omp_get_wtime();
   atb_par(A,B,C,Ni,Nj,Nk);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_par) mint_par=telapsed;
   if (telapsed > maxt_par) maxt_par=telapsed;
   for (int l = 0; l < Ni*Nj; l++) 
    if (fabs((C[l] - Cref[l])/Cref[l])>threshold) 
     {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", l, C[l], Cref[l]); return -1;}
  }
  printf("Best Performance (GFLOPS): ");
  printf("%.2f; ",2.0e-9*Ni*Nj*Nk/mint_par);
  printf("Worst Performance (GFLOPS): ");
  printf("%.2f ",2.0e-9*Ni*Nj*Nk/maxt_par);
  printf("\n");
}

