#!/bin/bash
#set max execution time
#PBS -l walltime=0:20:00

#set the execution queue
#PBS -q short_cpuQ
#PBS -o /home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi/job_output.out
#PBS -e /home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi/job_error.err
module load mpich-3.2  # Load the required module

echo "Running: Compiling SVD shared-memory version for all dataset sizes..."

project_root="/home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi"



# Compilation output directory and log directory
output_dir="$project_root/bin"
log_dir="$project_root/logs/compile"
code_dir="$project_root/src/SVD"
mkdir -p "$output_dir"
mkdir -p "$log_dir"

# Define datasets with their parameters (dataset identifier, number of users, number of movies)
datasets=(
    "1M 91054 67"
    "2M 104644 108"
    "4M 117773 224"
    "6M 129095 348"
    "8M 132588 465"
    "16M 139723 906"
    "32M 142255 1321"
)

# Loop over each dataset and compile a specialized executable.
for entry in "${datasets[@]}"; do
    read ds users movies <<< "$entry"
    gcc -fopenmp -std=c99 -o "$output_dir/svd_shared_${ds}" \
         "$code_dir/multi_threaded.c" "$code_dir/svds.c" "$code_dir/matrix_funcs.c" \
         -I/apps/OpenBLAS-0.3.7/include \
         -I/apps/lapack-3.8.0/include \
         -L/apps/OpenBLAS-0.3.7/lib \
         -L/apps/lapack-3.8.0/lib \
         -lopenblas -llapacke -lm \
         > "$log_dir/svd_shared_${ds}.out" 2> "$log_dir/svd_shared_${ds}.err"
    echo "Compiled binary for dataset ${ds} at: $output_dir/svd_shared_${ds}"
done

echo "All executables compiled successfully and stored in $output_dir."
