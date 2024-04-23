#! /bin/bash
  
######## Part 1 #########
# Script parameters     #
#########################
 
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
 
# Specify the QOS, mandatory option
#SBATCH --qos=normal
 
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=higgsgpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=firstjob
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=2

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/cepc/higgsgpu/wangzebing/ABCDnn/job-%j.out
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000
  
# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:2
    
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
########################################

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/publicfs/cms/user/wangzebing/anaconda/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/publicfs/cms/user/wangzebing/anaconda/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/publicfs/cms/user/wangzebing/anaconda/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/publicfs/cms/user/wangzebing/anaconda/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate hzgenv

date
#time(python DNN-CC.py)
cd /hpcfs/cepc/higgsgpu/wangzebing/ABCDnn/ABCDnn
time(python run_hzgABCDnn.py --batchsize $1 --depth $2 --nafdim $3 --LRrange3_0 $4 --LRrange3_1 $5 --LRrange3_2 $6 --permute $7)
##########################################
# Work load end

# Do not remove below this line

# list the allocated hosts
srun -l hostname
  
# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
  
sleep 180 

