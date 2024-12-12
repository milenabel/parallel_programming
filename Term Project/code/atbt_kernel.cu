__global__ void atbt_kernel(const float *A, const float *B, float *C, int Ni, int Nj, int Nk) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
float sum;

sum=0.0;
for (int k=0;k<Nk;k++)
// C[i][j] += A[k][i]*B[j][k];	  
   sum += A[i+Ni*k]*B[j*Nk+k];
C[i*Nj+j] = sum;
}

