#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=0:40:00
#PBS -q short_cpuQ
#PBS -N svd_shared_job

# Output and error file paths
#PBS -o job_output.out  
#PBS -e job_error.err   

# Load required modules (adjust versions if needed)
module load lapack-3.8.0
module load OpenBLAS-0.3.7
module load mpich-3.2 
# Set the number of OpenMP threads to use
export OMP_NUM_THREADS=2 
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Move to the working directory (where you submitted the job)
cd $PBS_O_WORKDIR

# Compile the code using a shared-memory (OpenMP) compiler
# Note: You might switch from mpicc to gcc if you are not using MPI calls anymore.
gcc -fopenmp -o svd_shared_16M_2 multi_threaded.c svds.c matrix_funcs.c \
  -I/apps/OpenBLAS-0.3.7/include \
  -I/apps/lapack-3.8.0/include \
  -L/apps/OpenBLAS-0.3.7/lib \
  -L/apps/lapack-3.8.0/lib \
  -lopenblas -llapacke -lm

# Run the executable (no need for mpirun if using pure OpenMP)
./svd_shared_16M_2 mapped_merged_data_16M.csv 139723 906 100
