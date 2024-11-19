#!/bin/bash -x
#SBATCH -M lonepeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=lonepeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C c20
#SBATCH -c 20
#SBATCH --exclusive
#SBATCH -t 0:05:00
#SBATCH --exclude lp[076-082]

module load clang

# Compile and run Problem 1
echo "Running Problem 1..."
clang -O3 -fopenmp -Rpass-missed=loop-vectorize -Rpass=loop-vectorize vec1_main.c vec1a.c vec1b.c vec1c.c -o vec1
srun ./vec1 > vec1_results.txt

# Compile and run Problem 2
echo "Running Problem 2..."
clang -O3 -fopenmp -Rpass-missed=loop-vectorize -Rpass=loop-vectorize vec2_main.c vec2_ref.c vec2_opt.c -o vec2
srun ./vec2 > vec2_results.txt

# Compile and run Problem 3
echo "Running Problem 3..."
clang -O3 -fopenmp -Rpass-missed=loop-vectorize -Rpass=loop-vectorize vec3_main.c vec3_ref.c vec3_opt.c -o vec3
srun ./vec3 > vec3_results.txt

# Notify completion
echo "All tasks completed. Check the output files for results."