void vec3_opt(int n, int Reps, float *__restrict__ w, float *__restrict__ x, float *__restrict__ y) {
  int rep, i;

  for (rep = 0; rep < Reps; rep++) {
      // First, update w[i] independently
      for (i = 0; i < n - 1; i++) {
          w[i] = y[i] + 1;
      }

      // Then, update y[i + 1] independently
      for (i = 0; i < n - 1; i++) {
          y[i + 1] = 2 * x[i];
      }
  }
}