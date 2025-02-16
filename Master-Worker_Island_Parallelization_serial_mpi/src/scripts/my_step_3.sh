#!/bin/bash
#PBS -l walltime=0:20:00
#PBS -q short_cpuQ
module load mpich-3.2  
#PBS -o /home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi/job_output.out
#PBS -e /home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi/job_error.err
# Base directories (adjust these as needed)
project_root="/home/arbaaz.ali/project/HPC_uni-tn_24/Master-Worker_Island_Parallelization_serial_mpi"
scripts_dir="$project_root/scripts"   
log_dir="$project_root/logs"                 

# List of dataset identifiers (should match what was used in Step 2)
DATASETS=("1M" "2M" "4M" "6M" "8M" "16M" "32M")

echo "Submitting all run-job scripts..."

# Loop over each dataset folder
for ds in "${DATASETS[@]}"; do
    dataset_script_dir="$scripts_dir/$ds"
    
    # Submit the serial run script (1_core.sh)
    if [ -f "$dataset_script_dir/1_core.sh" ]; then
        echo "Submitting serial job for dataset $ds"
        qsub "$dataset_script_dir/1_core.sh"
    else
        echo "Serial job script for dataset $ds not found!"
    fi
    
    # Submit all parallel run scripts in this dataset folder.
    # These files are named like "2_cores.sh", "4_cores.sh", etc.
    for job_script in "$dataset_script_dir/"*_cores.sh; do
        if [ -f "$job_script" ]; then
            echo "Submitting parallel job: $job_script"
            qsub "$job_script"
        else
            echo "Parallel job script $job_script not found!"
        fi
    done
done

echo "All run jobs submitted. Please check the logs in $log_dir for output and error messages."
