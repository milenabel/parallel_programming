// Use "clang -O3 -fopenmp -Rpass-missed=loop-vectorize -Rpass=loop-vectorize " to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#define NTrials 5
#define NReps 1000
#define N 1024

void vec1a(int size, int Reps, float *__restrict__ x);
void vec1b(int size, int Reps, float *__restrict__ x);
void vec1c(int size, int Reps, float *__restrict__ x);

float x[N];

int main(int argc, char *argv[]){
  double tstart,telapsed;
  
  int i,j,k,nt,trial,num_cases;
  double mint,maxt;
  
  printf("Matrix Size = %d; NTrials=%d\n",N,NTrials);
  
  for(i=0;i<N;i++)
    x[i] = 10.0*i/sqrt(N);
  
  vec1a(N, NReps, &x[0]);
   
  printf("Performance (GFLOPS) for stmt 'w[i] = w[i]+1':");
  mint = 1e9; maxt = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = omp_get_wtime();
   vec1a(N, NReps, &x[0]);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint) mint=telapsed;
   if (telapsed > maxt) maxt=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N*NReps/maxt,1.0e-9*N*NReps/mint);

  printf("Performance (GFLOPS) for stmt 'w[i] = w[i+1]+1':");
  mint = 1e9; maxt = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = omp_get_wtime();
   vec1b(N, NReps, &x[0]);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint) mint=telapsed;
   if (telapsed > maxt) maxt=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N*NReps/maxt,1.0e-9*N*NReps/mint);

  printf("Performance (GFLOPS) for stmt 'w[i] = w[i-1]+1':");
  mint = 1e9; maxt = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   tstart = omp_get_wtime();
   vec1c(N, NReps, &x[0]);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint) mint=telapsed;
   if (telapsed > maxt) maxt=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",1.0e-9*N*NReps/maxt,1.0e-9*N*NReps/mint);
} 
