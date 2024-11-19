// Use "clang -O3 -fopenmp -Rpass-missed=loop-vectorize -Rpass=loop-vectorize " to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#define NTrials 5
#define N 1024
#define NREPS 100
#define threshold (0.0000001)

void compare(int n, float ref[n], float test[n]);
void vec5_ref(int n, float *__restrict__ A, float *__restrict__ x);
void vec5_opt(int n, float *__restrict__ A, float *__restrict__ x);

float a[N][N], xref[N], x[N];

int main(int argc, char *argv[]){
  double tstart,telapsed;
  
  int i,j,k,nt,trial,num_cases;
  double mint_opt,maxt_opt;
  double mint_ref,maxt_ref;
  
  printf("Matrix Size = %d; NTrials=%d\n",N,NTrials);
  
  for(i=0;i<N;i++)
   for(j=0;j<N;j++)
    a[i][j] = 1.1*(2*i+j);
  
// Warmup instance is not timed
  for(i=0;i<N;i++) xref[i] = 0.0;
  vec5_ref(N, &a[0][0], &xref[0]);
  printf("Reference code performance in GFLOPS");
  mint_ref = 1e9; maxt_ref = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<N;i++) xref[i] = 0.0;
   tstart = omp_get_wtime();
   vec5_ref(N, &a[0][0], &xref[0]);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_ref) mint_ref=telapsed;
   if (telapsed > maxt_ref) maxt_ref=telapsed;
  }
   printf(" Min: %.2f; Max: %.2f\n",2.0e-9*N*N/maxt_ref,2.0e-9*N*N/mint_ref);
  
   mint_opt = 1e9; maxt_opt = 0;
   for (trial=0;trial<NTrials;trial++)
   {
    for(i=0;i<N;i++) x[i] = 0;
    tstart = omp_get_wtime();
    vec5_opt(N, &a[0][0], &x[0]);
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_opt) mint_opt=telapsed;
    if (telapsed > maxt_opt) maxt_opt=telapsed;
    compare(N,xref,x);
   }
  printf("Optimized code performance in GFLOPS");
  printf(" Min: %.2f; Max: %.2f\n",2.0e-9*N*N/maxt_opt,2.0e-9*N*N/mint_opt);
}

void compare(int n, float ref[n], float test[n])
{
  float maxdiff,this_diff;
  int numdiffs;
  int i,j;
  numdiffs = 0;
  maxdiff = 0;
  for (i=0;i<n;i++)
      {
        this_diff = ref[i]-test[i];
        if (this_diff < 0) this_diff = -1.0*this_diff;
        if (this_diff>threshold)
          { numdiffs++;
            if (this_diff > maxdiff) maxdiff=this_diff;
          }
      }
  if (numdiffs > 0)
  { printf("Error : %d Differences found over threshold %f; Max Diff = %f\n",
           numdiffs,threshold,maxdiff);
    printf("Exiting\n"); exit(-1);
  
  }
}
