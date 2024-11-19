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
#SBATCH --output=trmm_permuted_output.%j.txt

echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log
echo " "

# Test Dynamic Scheduling, Chunk Size 1
cp trmm_par_guided_10.c trmm_par.c
gcc -O3 -fopenmp -o trmm_permuted trmm_main.c trmm_ref.c trmm_par.c
echo "Trmm JIK Loop - Guided Scheduling, Chunk Size 10: Trial 1" | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log
./trmm_permuted | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log

# Test Dynamic Scheduling, Chunk Size 5
cp trmm_par_guided_20.c trmm_par.c
gcc -O3 -fopenmp -o trmm_permuted trmm_main.c trmm_ref.c trmm_par.c
echo "Trmm JIK Loop - Guided Scheduling, Chunk Size 20: Trial 1" | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log
./trmm_permuted | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log

# Test Dynamic Scheduling, Chunk Size 10
cp trmm_par_guided_40.c trmm_par.c
gcc -O3 -fopenmp -o trmm_permuted trmm_main.c trmm_ref.c trmm_par.c
echo "Trmm JIK Loop - Guided Scheduling, Chunk Size 40: Trial 1" | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log
./trmm_permuted | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log

# Test Guided Scheduling
cp trmm_par_guided.c trmm_par.c
gcc -O3 -fopenmp -o trmm_permuted trmm_main.c trmm_ref.c trmm_par.c
echo "Trmm JIK Loop - Guided Scheduling: Trial 1" | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log
./trmm_permuted | tee -a lonepeak_trmm_permuted.$SLURM_JOB_ID\.log
