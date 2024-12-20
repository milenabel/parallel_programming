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
echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_trmm.$SLURM_JOB_ID\.log
echo " "
gcc -O3 -fopenmp -o trmm trmm_main.c trmm_ref.c trmm_par.c
echo "Trmm trial 1"
./trmm | tee -a lonepeak_trmm.$SLURM_JOB_ID\.log
echo "Trmm trial 2"
./trmm | tee -a lonepeak_trmm.$SLURM_JOB_ID\.log
echo "Trmm trial 3"
./trmm | tee -a lonepeak_trmm.$SLURM_JOB_ID\.log
