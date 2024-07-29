#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l h_rt=240:0:0
#$ -l h_vmem=5G
#$ -t 1-14

module load trimgalore

INPUT_FILE=$(sed -n "${SGE_TASK_ID}p" samples2.txt)

trim_galore --paired --rrbs  --fastqc -j 8 $INPUT_FILE -o trimming

