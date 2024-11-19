#!/bin/bash -x
#SBATCH -M kingspeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=kingspeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH -t 0:05:00
echo "*** Assigned Kingspeak Node: " $SLURMD_NODENAME | tee -a kingspeak_trmm.$SLURM_JOB_ID\.log
echo " "
gcc -O3 -fopenmp -o trmm trmm_main.c trmm_ref.c trmm_par.c
echo "Trmm trial 1"
./trmm | tee -a kingspeak_trmm.$SLURM_JOB_ID\.log
echo "Trmm trial 2"
./trmm | tee -a kingspeak_trmm.$SLURM_JOB_ID\.log
echo "Trmm trial 3"
./trmm | tee -a kingspeak_trmm.$SLURM_JOB_ID\.log
