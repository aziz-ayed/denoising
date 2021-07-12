#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M aziz.ayed@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N shapepipe_smp
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=10:00:00
# Request number of cores
#PBS -l nodes=1:ppn=2
# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"
export CONFDIR="$HOME/scripts_datasets"
# Activate conda environment
module load intelpython/3-2020.1
module load intel/19.0/2
source activate shapepipe
# Run ShapePipe using full paths to executables
## $SPENV/bin/shapepipe_run -c $CONFDIR/mccd_test14.ini
python $CONFDIR/generate_datasets.py --catalog_bin 0
# Return exit code
exit 0
