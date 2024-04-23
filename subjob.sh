#! /bin/bash

#sbatch runjob.sh 12288 2 128 0.001 0.000001 20000 0 # good
#sbatch runjob.sh 12288 4 128 0.001 0.000001 20000 0 # good
#sbatch runjob.sh 12288 8 128 0.001 0.000001 20000 0 # good

#sbatch runjob.sh 4096 2 128 0.001 0.000001 20000 0 # good
#sbatch runjob.sh 4096 4 128 0.001 0.000001 20000 0 # good
#sbatch runjob.sh 4096 8 128 0.001 0.000001 20000 0 # good

#sbatch runjob.sh 4096 2 128 0.01 0.00001 20000 0 # good
#sbatch runjob.sh 4096 4 128 0.01 0.00001 20000 0 # good
#sbatch runjob.sh 4096 8 128 0.01 0.00001 20000 0 # good

#sbatch runjob.sh 7900 2 128 0.001 0.000001 20000 0 # good ########
#sbatch runjob.sh 7900 4 128 0.001 0.000001 20000 0 # good
#sbatch runjob.sh 7900 8 128 0.001 0.000001 20000 0 # good

#sbatch runjob.sh 7900 2 128 0.001 0.00001 10000 0 # good
#sbatch runjob.sh 7900 4 128 0.001 0.00001 10000 0 # good
#sbatch runjob.sh 7900 8 128 0.001 0.00001 10000 0 # good

#sbatch runjob.sh 7900 2 128 0.001 0.001 20000 0 # good
#sbatch runjob.sh 7900 4 128 0.001 0.001 20000 0 # good
#sbatch runjob.sh 7900 8 128 0.001 0.001 20000 0 # good

#sbatch runjob.sh 7900 2 128 0.0001 0.0001 20000 0 # good
#sbatch runjob.sh 7900 4 128 0.0001 0.0001 20000 0 # good
#sbatch runjob.sh 7900 8 128 0.0001 0.0001 20000 0 # good ##############
#sbatch runjob.sh 7900 16 128 0.0001 0.0001 20000 0 # good
#sbatch runjob.sh 7900 2 128 0.0001 0.0001 20000 1 # good
#sbatch runjob.sh 7900 4 128 0.0001 0.0001 20000 1 # good ############
#sbatch runjob.sh 7900 8 128 0.0001 0.0001 20000 1 # good ############
#sbatch runjob.sh 7900 16 128 0.0001 0.0001 20000 1 # good




sbatch runjob.sh 2048 16 128 0.0001 0.0001 20000 1 # good
sbatch runjob.sh 2048 16 128 0.0001 0.00001 20000 1 # good 1
sbatch runjob.sh 7900 16 128 0.0001 0.0001 20000 1 # good
sbatch runjob.sh 7900 16 128 0.0001 0.000001 20000 1 # good 2
sbatch runjob.sh 7900 16 128 0.0001 0.000001 2000 1 # good