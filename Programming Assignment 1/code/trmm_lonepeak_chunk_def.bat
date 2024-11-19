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
#SBATCH --output=trmm_chunk_def_output.%j.txt

echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_trmm_chunk_def.$SLURM_JOB_ID\.log
echo " "
module load gcc
gcc -O3 -fopenmp -o trmm_chunk_def trmm_main.c trmm_ref.c trmm_par_outer_chunk_def.c
echo "Trmm Default Chunk Size: Trial 1"
./trmm_chunk_def | tee -a lonepeak_trmm_chunk_def.$SLURM_JOB_ID\.log