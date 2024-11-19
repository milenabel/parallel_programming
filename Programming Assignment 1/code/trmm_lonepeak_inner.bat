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
#SBATCH --output=trmm_inner_output.%j.txt

echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_trmm_inner.$SLURM_JOB_ID\.log
echo " "
gcc -O3 -fopenmp -o trmm_inner trmm_main.c trmm_ref.c trmm_par_inner.c
echo "Trmm Inner Loop Version: Trial 1"
./trmm_inner | tee -a lonepeak_trmm_inner.$SLURM_JOB_ID\.log
echo "Trmm Inner Loop Version: Trial 2"
./trmm_inner | tee -a lonepeak_trmm_inner.$SLURM_JOB_ID\.log
echo "Trmm Inner Loop Version: Trial 3"
./trmm_inner | tee -a lonepeak_trmm_inner.$SLURM_JOB_ID\.log