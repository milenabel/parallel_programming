#include <stdio.h>
#include <time.h>
#define Ntrials 5
#define threshold 0.00001

void checkCUDAError(const char *msg);


void atbt_launch(const float *d_A, const float *d_B, float *d_C, int Ni, int Nj, int Nk);

int main(){

  cudaEvent_t start, stop;
  float elapsedTime, tmin, tmax;
  float *sum, *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k,Ni,Nj,Nk;

  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  sum  = (float *) malloc(sizeof(float)*Ni);
  h_A = (float *) malloc(sizeof(float)*Nk*Ni);
  h_B = (float *) malloc(sizeof(float)*Nj*Nk);
  h_C = (float *) malloc(sizeof(float)*Ni*Nj);
  h_Cref = (float *) malloc(sizeof(float)*Ni*Nj);
  for (k=0; k<Nk; k++)
   for (i=0; i<Ni; i++)
    h_A[k*Ni+i] = k*Ni+i-1;
  for (j=0; j<Nj; j++)
   for (k=0; k<Nk; k++)
    h_B[j*Nk+k] = j*Nk+k+1;
  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++) {
    h_C[i*Nj+j] = 0;
    h_Cref[i*Nj+j] = 0;}

  for (j=0;j<Nj;j++)
  {
   for (i=0;i<Ni;i++) sum[i]=0;
   for (k=0;k<Nk;k++)
    for (i=0;i<Ni;i++)
// h_Cref[i][j] += h_A[k][i]*h_B[j][k];
     sum[i] += h_A[k*Ni+i]*h_B[j*Nk+k];
   for (i=0;i<Ni;i++) h_Cref[i*Nj+j] = sum[i];
  }
// Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Nk*Ni*sizeof(float));
  cudaMalloc(&d_B, Nj*Nk*sizeof(float));
  cudaMalloc(&d_C, Ni*Nj*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, Nk*Ni*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nj*Nk*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D failure");

  tmin = 1e9; tmax = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for(int trial=0;trial<Ntrials;trial++)
  {
   cudaEventRecord(start);
   // Launch kernel
   atbt_launch(d_A, d_B, d_C, Ni, Nj, Nk);
   cudaEventRecord(stop);
   checkCUDAError("kernel launch");
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
   if (elapsedTime < tmin) tmin=elapsedTime;
   if (elapsedTime > tmax) tmax=elapsedTime;
   // Copy results back to host
   cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(float), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H");
   for (int l = 0; l < Ni*Nj; l++) if (fabs((h_C[l] - h_Cref[l])/h_Cref[l])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", l, h_C[l], h_Cref[l]); return -1;}
  }
  printf("AtBt <Ni=%d,Nj=%d,Nk=%d>: Over %d trials, Min_GFLOPS: %.2f; Max_GFLOPS: %2f\n",Ni,Nj,Nk,Ntrials,2.0e-6*Ni*Nj*Nk/tmax,2.0e-6*Ni*Nj*Nk/tmin);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}


