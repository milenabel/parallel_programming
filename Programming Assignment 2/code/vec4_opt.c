void vec4_opt(int n, int Reps, float *__restrict__ w, float *__restrict__ x, float *__restrict__ y)
{
int rep,i;

 for(rep=0;rep<Reps;rep++)
  {
   for(i=0;i<n-1;i++)
    {
      w[i+1] = y[i]+1;
      y[i+1] = x[i]+w[i];
    }
  }
}
