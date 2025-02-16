#!/bin/bash
#PBS -l walltime=0:20:00
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Configuration
BASE_DIR="/home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi/scripts"  # Root directory where the scripts are generated
DATASETS=(
    "1M 91054 67"
    "2M 104644 108"
    "4M 117773 224"
    "6M 129095 348"
    "8M 132588 465"
    "16M 139723 906"
    "32M 142255 1321"
)  # Dataset identifier, number of users, number of movies
CORES_LIST=(2 4 8 16 32 64)  # Number of cores to use
K_VALUE=100  # This can be dynamic, or keep a constant K value for SVD factorization

# Serialization format
generate_serial_script() {
    local dataset=$1
    local users=$2
    local movies=$3
    local output_dir="$BASE_DIR/$dataset"
    mkdir -p "$output_dir"

    cat <<EOF > "$output_dir/1_core.sh"
#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=1:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

# Path
project_root="/home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi"
bin_dir="\$project_root/bin"
log_dir="\$project_root/logs/serial/${dataset}" 
mkdir -p \$log_dir

# Dataset and parameters
dataset=${dataset}
users=${users}
movies=${movies}
K=${K_VALUE}

echo "Running serial version for dataset: \$dataset with \$users users and \$movies movies (K=\$K)..."
mpirun.actual -n 1 "\$bin_dir/svd_shared_\${dataset}" \$dataset \$users \$movies \$K \
    > "\$log_dir/output.log" 2> "\$log_dir/error.log"
EOF
    echo "Generated: $output_dir/1_core.sh"
}

# Parallelization format
generate_parallel_script() {
    local dataset=$1
    local users=$2
    local movies=$3
    local cores=$4
    local output_dir="$BASE_DIR/$dataset"
    mkdir -p "$output_dir"

    cat <<EOF > "$output_dir/${cores}_cores.sh"
#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=${cores}:mem=2gb -l place=pack:excl
#PBS -q short_cpuQ
module load mpich-3.2 # Load the required module

export OMP_NUM_THREADS=${cores}
echo "Using OMP_NUM_THREADS=${cores}"

# Path
project_root="/home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi"
bin_dir="\$project_root/bin"
log_dir="\$project_root/logs/parallel/${dataset}/${cores}"  
mkdir -p \$log_dir

# Dataset and parameters
dataset=${dataset}
users=${users}
movies=${movies}
cores=${cores}
K=${K_VALUE}

echo "Running parallel (MPI) version for dataset: \$dataset with \$users users, \$movies movies (K=\$K) using \$cores cores..."
mpirun.actual -n ${cores} "\$bin_dir/svd_shared_\${dataset}" \$bin_dir/mapped_merged_data_\${dataset}.csv \$users \$movies \$K \
    > "\$log_dir/output.log" 2> "\$log_dir/error.log"
EOF
    echo "Generated: $output_dir/${cores}_cores.sh"
}

# Main function to loop through datasets and generate scripts
main() {
    for entry in "${DATASETS[@]}"; do
        read dataset users movies <<< "$entry"

        # Generate serial script
        generate_serial_script "$dataset" "$users" "$movies"

        # Generate parallel scripts for different core counts
        for cores in "${CORES_LIST[@]}"; do
            generate_parallel_script "$dataset" "$users" "$movies" "$cores"
        done
    done
}
main
