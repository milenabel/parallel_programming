void vec1a (int size, int Reps, float *__restrict__ w)
{
int rep,i;

 for(rep=0;rep<Reps;rep++)
  {
   for(i=1;i<size-1;i++) w[i] = w[i]+1; 
  }
}
