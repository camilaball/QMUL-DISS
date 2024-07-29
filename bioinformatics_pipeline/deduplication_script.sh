#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l h_rt=240:0:0
#$ -l h_vmem=5G
#$ -m beas
#$ -t 1-x
#$ -tc x

module load samtools
module load bismark

INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" bam_files.txt)

deduplicate_bismark -p --bam $INPUT_FILE --output_dir deduplication
