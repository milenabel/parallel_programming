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
#SBATCH --output=trmm_outer_output.%j.txt

echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_trmm_outer.$SLURM_JOB_ID\.log
echo " "
gcc -O3 -fopenmp -o trmm_outer trmm_main.c trmm_ref.c trmm_par_outer.c
echo "Trmm Outer Loop Version: Trial 1"
./trmm_outer | tee -a lonepeak_trmm_outer.$SLURM_JOB_ID\.log
echo "Trmm Outer Loop Version: Trial 2"
./trmm_outer | tee -a lonepeak_trmm_outer.$SLURM_JOB_ID\.log
echo "Trmm Outer Loop Version: Trial 3"
./trmm_outer | tee -a lonepeak_trmm_outer.$SLURM_JOB_ID\.log