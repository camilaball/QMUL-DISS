#!/bin/bash 
#$ -cwd 
#$ -pe smp 8 
#$ -l h_rt=240:0:0 
#$ -l h_vmem=5G 
#$ -m beas 
#$ -t 1-14 
#$ -tc 6


module load samtools  

module load bismark 

  

INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" extraction_samples.txt) 

  

bismark_methylation_extractor --gzip --bedGraph --paired-end --multicore 4 $INPUT_FILE  -o methylation_extraction 
