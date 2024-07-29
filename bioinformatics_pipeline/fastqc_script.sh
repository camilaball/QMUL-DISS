#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l h_rt=240:0:0
#$ -l h_vmem=5G
#$ -m beas

module load fastqc

fastqc -o results_fastqc SRR*
 
