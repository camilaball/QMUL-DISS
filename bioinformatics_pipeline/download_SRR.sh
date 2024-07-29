#!/bin/bash 
#$ -cwd 
#$ -pe smp 8 
#$ -l h_rt=240:0:0 
#$ -l h_vmem=5G 
#$ -m beas 
#$ -t 1-14 
#$ -tc 5

module load sratools 

INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" samples.txt) 
#samples.txt contaings the SRR runs of all the files needed  

fastq-dump --gzip  --split-files $INPUT_FILE -O concat_tryout
