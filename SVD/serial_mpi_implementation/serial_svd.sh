#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=0:20:00
#PBS -q short_cpuQ
#PBS -N svd_serial_job
#PBS -o job_output4M.out  
#PBS -e job_error4M.err 

# Load required modules (adjust versions if needed)
module load lapack-3.8.0
module load OpenBLAS-0.3.7
module load mpich-3.2

# Move to the working directory (where you submitted the job)
cd $PBS_O_WORKDIR

# Compile the code
# -I. adds current directory to the include path (if svds.h or matrix_funcs.h are in current dir)
mpicc -o svd_mpi_serial_32M main.c svds.c matrix_funcs.c \
  -I/apps/OpenBLAS-0.3.7/include \
  -I/apps/lapack-3.8.0/include \
  -L/apps/OpenBLAS-0.3.7/lib \
  -L/apps/lapack-3.8.0/lib \
  -lopenblas -llapacke -lm

# If your code expects command-line arguments, list them after the executable.
# For instance, if your 'main.c' expects: ./svd_mpi_serial <csv_file> <num_users> <num_movies>
# Adjust these arguments to match your scenario. Example:
MAPPED_DATA="mapped_merged_data_32M.csv"
NUM_USERS=142255
NUM_MOVIES=1321
K_value=100
# Run the compiled executable with 1 MPI process (serial)
mpirun.actual -np 1 ./svd_mpi_serial_32M $MAPPED_DATA $NUM_USERS $NUM_MOVIES $K_value

# You can also run simply:
# ./svd_mpi_serial_2M $MAPPED_DATA $NUM_USERS $NUM_MOVIES
# depending on your HPC environment. Some systems require mpirun or mpiexec explicitly.
