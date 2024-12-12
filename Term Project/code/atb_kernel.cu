__global__ void atb_kernel(const float *A, const float *B, float *C, int Ni, int Nj, int Nk) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
float sum;

sum=0.0;
for (int k=0;k<Nk;k++)
// C[i][j] += A[k][i]*B[k][j];	  
   sum += A[i+Ni*k]*B[k*Nj+j];
C[i*Nj+j] = sum;
}

