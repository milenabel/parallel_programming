#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define threshold 0.0000001

void mmvseq(int n, int m, double a[][n], double x[][m], double y[][m]);
void mmvpar(int n, int m, double a[][n], double x[][m], double y[][m]);
void compare(int n, int m, double wref[][m], double w[][m]);

double A[1024][1024], X[1024][16],Temp[1024][16],XX[1024][16],Temp1[1024][16];
int myid, nprocs;
int main(int argc, char *argv[]) {

double clkbegin, clkend;
double t, tmax, *tarr;
int i,j,it;

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

  for(i=0;i<1024;i++)
   { 
     for(j=0;j<1024;j++) A[i][j] = 2.0*((i+j) % 1024)/(1.0*1024*1023);
     for(j=0;j<16;j++) X[i][j] = XX[i][j] = sqrt(1.0*(i+j));
   }

  if (myid == 0) 
  {
   clkbegin = MPI_Wtime();
   mmvseq(1024,16,A,X,Temp);
   clkend = MPI_Wtime();
   t = clkend-clkbegin;
   printf("Repeated MMV: Sequential Version: %.2f GFLOPS; Time = %.3f sec; \n",
           2.0*1e-9*1024*1024*16*10/t,t);

  }

  MPI_Barrier(MPI_COMM_WORLD);

  clkbegin = MPI_Wtime();
  mmvpar(1024,16,A,XX,Temp1);
  clkend = MPI_Wtime();
  t = clkend-clkbegin;
  MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
  if (myid == 0)
  {
   printf("Repeated MMV: Parallel Version: %.2f GFLOPS; Time = %.3f sec; \n",
          2.0*1e-9*1024*1024*16*10/tmax,tmax);
   compare(1024,16,X,XX);
  }
  MPI_Finalize();
}


void mmvseq(int n, int m, double a[][n], double x[][m], double y[][m])
{ int i,j,k,iter;
  
  for (i=0;i<n;i++) for (j=0;j<m;j++) y[i][j]=0; 
  for(iter=0;iter<10;iter++)
   {
    for(i=0;i<n;i++)
     for(k=0;k<n;k++) 
      for(j=0;j<m;j++)
       y[i][j] += a[i][k]*x[k][j];
    for (i=0; i<n; i++) for (j=0;j<m;j++) x[i][j] = sqrt(y[i][j]);
   }
}

void mmvpar(int n, int m, double a[][n], double x[][m], double y[][m])
// FIXME: Initially identical to reference; make your changes to parallelize this code
{ int i,j,k,iter;
  
  for (i=0;i<n;i++) for (j=0;j<m;j++) y[i][j]=0; 
  for(iter=0;iter<10;iter++)
   {
    for(i=0;i<n;i++)
     for(k=0;k<n;k++) 
      for(j=0;j<m;j++)
       y[i][j] += a[i][k]*x[k][j];
    for (i=0; i<n; i++) for (j=0;j<m;j++) x[i][j] = sqrt(y[i][j]);
   }
}

void compare(int n, int m, double wref[][m], double w[][m])
{
double maxdiff,this_diff;
double minw,maxw,minref,maxref;
int numdiffs;
int i,j;
  numdiffs = 0;
  maxdiff = 0;
  minw = minref = 1.0e9;
  maxw = maxref = -1.0;
  for (i=0;i<n;i++) for (j=0;j<m;j++)
    {
     this_diff = wref[i][j]-w[i][j];
     if (w[i][j] < minw) minw = w[i][j];
     if (w[i][j] > maxw) maxw = w[i][j];
     if (wref[i][j] < minref) minref = wref[i][j];
     if (wref[i][j] > maxref) maxref = wref[i][j];
     if (this_diff < 0) this_diff = -1.0*this_diff;
     if (this_diff>threshold)
      { numdiffs++;
        if (this_diff > maxdiff) maxdiff=this_diff;
      }
    }
   if (numdiffs > 0)
      printf("%d Diffs found over threshold %f; Max Diff = %f\n",
               numdiffs,threshold,maxdiff);
   else
      printf("No differences found between base and test versions\n");
   printf("MinRef = %f; MinPar = %f; MaxRef = %f; MaxPar = %f\n",
          minref,minw,maxref,maxw);
}

