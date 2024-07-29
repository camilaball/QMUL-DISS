#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l h_rt=240:0:0
#$ -l h_vmem=5G
#$ -t 1-14

module load bismark

INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" samples2.txt)

/data/home/bt23801/.conda/envs/fastq_screen/share/fastq-screen-0.15.3-0/fastq_screen --bisulfite $INPUT_FILE -o results_fastqc_screen

