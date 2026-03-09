#!/bin/bash

#SBATCH -A your_account
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --job-name=store
#SBATCH --out=out/store.log
#SBATCH --err=out/store.log

module load spack
spack load gcc

# 1. Force the linker to use the Spack GCC libraries
export GCC_PATH=$(spack location -i gcc)
export LD_LIBRARY_PATH=$GCC_PATH/lib64:$GCC_PATH/lib:$LD_LIBRARY_PATH

# 2. Environment Variables
export HF_HOME="data_/hf"
export HF_DATASETS_OFFLINE="1"
export HF_DATASETS_CACHE="data_/hf/datasets"

# 3. Start Qdrant from its own directory using a subshell
# Parentheses ( ) create a subshell so the 'cd' doesn't affect the rest of this script
(
    cd ../libs/qdrant || exit
    ./target/release/qdrant &
)

# 4. Give Qdrant time to warm up
sleep 120

# 5. Run the Python modules in parallel
# Note: srun -u is good for real-time logs
srun -u --exclusive --ntasks=1 python -m artseek.data.main make-qdrant-store --process-idx 0 --num-proc 4 &
srun -u --exclusive --ntasks=1 python -m artseek.data.main make-qdrant-store --process-idx 1 --num-proc 4 &
srun -u --exclusive --ntasks=1 python -m artseek.data.main make-qdrant-store --process-idx 2 --num-proc 4 &
srun -u --exclusive --ntasks=1 python -m artseek.data.main make-qdrant-store --process-idx 3 --num-proc 4 &

# 6. Wait for all 4 Python tasks to finish
# We don't use -n here because we need ALL of them to complete
wait %2 %3 %4 %5 2>/dev/null || wait

# 7. Cleanly shut down Qdrant
echo "Tasks finished. Shutting down Qdrant..."
pkill qdrant
wait
