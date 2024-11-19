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

# Compile and run Problem 5
echo "Running Problem 5..."
clang -O3 -fopenmp -Rpass-missed=loop-vectorize -Rpass=loop-vectorize vec1_main.c vec1a.c vec1b.c vec1c.c -o vec1
clang -O3 -fopenmp -mavx -Rpass-missed=loop-vectorize -Rpass=loop-vectorize vec5_main.c vec5_ref.c vec5_opt.c -o vec5
srun ./vec5 > vec5_results.txt

# Notify completion
echo "All tasks completed. Check the output files for results."