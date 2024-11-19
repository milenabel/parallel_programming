#!/bin/bash -x
#SBATCH -M lonepeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=lonepeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C c20
#SBATCH -c 20
#SBATCH --exclusive
#SBATCH -t 0:10:00
#SBATCH --exclude lp[076-082]
#SBATCH --output=trmm_unrolled_output.%j.txt

echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_trmm_unrolled.$SLURM_JOB_ID\.log
echo " "

# Compile and run the unrolled version
cp trmm_par_unrolled.c trmm_par.c
gcc -O3 -fopenmp -march=native -mtune=native -o trmm_unrolled trmm_main.c trmm_ref.c trmm_par.c

echo "Trmm JIK Loop - Guided Scheduling with Unrolling: Trial 1" | tee -a lonepeak_trmm_unrolled.$SLURM_JOB_ID\.log
./trmm_unrolled | tee -a lonepeak_trmm_unrolled.$SLURM_JOB_ID\.log
echo "Trmm JIK Loop - Guided Scheduling with Unrolling: Trial 2" | tee -a lonepeak_trmm_unrolled.$SLURM_JOB_ID\.log
./trmm_unrolled | tee -a lonepeak_trmm_unrolled.$SLURM_JOB_ID\.log
echo "Trmm JIK Loop - Guided Scheduling with Unrolling: Trial 3" | tee -a lonepeak_trmm_unrolled.$SLURM_JOB_ID\.log
