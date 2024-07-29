#!/bin/bash
#$ -cwd 
#$ -pe smp 8 
#$ -l h_rt=240:0:0 
#$ -l h_vmem=5G 
#$ -t 1-14
#$ -m beas
#$ -tc 4
module load bismark 
module load samtools 

INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" alignment_samples.txt) 
REPCORES=$((NSLOTS / 2)) 
 #The input file in this case would be the forward reads R1  

# Execute Bismark with the specified parameters 
 

bismark --genome /data/scratch/bt23801/genomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/ -p ${REPCORES}  -1 ${INPUT_FILE} -2 ${INPUT_FILE//_1_val_1/_2_val_2} -o alignment  


