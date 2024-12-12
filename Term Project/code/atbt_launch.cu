__global__ void atbt_kernel(const float *A, const float *B, float *C, int Ni, int Nj, int Nk);

void atbt_launch(const float *d_A, const float *d_B, float *d_C, int Ni, int Nj, int Nk)
{
  dim3 block(16,16);  
  dim3 grid(ceil(Ni/float(16)),ceil(Nj/float(16)));
  atbt_kernel<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
}
