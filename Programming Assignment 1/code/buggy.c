#include <omp.h>
#include <stdio.h>

#define numElems (128*1024*1024)
float src[numElems];
float dest[numElems], destS[numElems];

int main(int argc, char *argv[]){
  int i, numThreads, errors;
  double start_time, run_time;

  for (i = 0; i < numElems; i++) {
    src[i] = i;
    destS[i] = 0;
    dest[i] = 0;
  }

  // Serial version reference
  for (i = 0; i < numElems; ++i) destS[i] = src[i] + 1;

  numThreads = 4;
  printf("Setting number of threads in parallel region to: %d\n", numThreads);
  omp_set_num_threads(numThreads);

  // Corrected parallel version with a single loop
  #pragma omp parallel for
  for (i = 0; i < numElems; ++i) {
    dest[i] = src[i] + 1;
  }

  // Correctness check
  errors = 0;
  for (i = 0; i < numElems; i++) 
  {
    if(destS[i] != dest[i]) errors += 1;
  }

  if (errors ==0) printf("Correctness check passed for parallel version 2\n");
     else printf("Correctness check failed for parallel version; %d errors\n",errors);
}
