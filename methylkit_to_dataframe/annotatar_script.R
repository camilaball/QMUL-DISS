#if not installed 
#BiocManager::install("TxDb.Hsapiens.UCSC.hg38.knownGene") 
#BiocManager::install("AnnotationHub") 
#BiocManager::install("org.Hs.eg.db") 
#install.packages("ggplot2")

#Loading the right libraries
library(annotatr)
library(TxDb.Hsapiens.UCSC.hg38.knownGene) 
library(AnnotationHub) 
library(GenomicRanges)
library(ggplot2)

#Loading the methylation dataset 
methylation_data <- read.csv('/data/scratch/bt23801/rrbs/methylkit_tryout_blood/samples_beta_value.csv')

#create a GRanges object
gr <- GRanges(
  seqnames = methylation_data$chr,
  ranges = IRanges(start = methylation_data$start, end = methylation_data$end),
  strand = methylation_data$strand
)

#Select annotations 
annots = c('hg38_cpgs','hg38_basicgenes','hg38_genes_intergenic','hg38_genes_intronexonboundaries','hg38_cpg_islands','hg38_cpg_shores','hg38_cpg_shelves','hg38_cpg_inter')

#build the annotations 
annotations_granges=build_annotations(genome = 'hg38', annotations=annots)

#annotate methylated regions
annotated <- annotate_regions( 
  regions = gr, 
  annotations = annotations_granges, 
  ignore.strand = TRUE, 
  quiet = FALSE 
  ) 

#check it 
df_annoted=data.frame(annotated)


#write it as csv
write.csv(df_annoted,"annotated_RRBS_sample.csv",row.names=FALSE)

#view the number of regions per annotation. This function
# is useful when there is no classification or data
# associated with the regions.
annots_order = c(
  'hg38_genes_1to5kb',
  'hg38_genes_promoters',
  'hg38_genes_5UTRs',
  'hg19_genes_exons',
  'hg38_genes_introns',
  'hg38_genes_3UTRs',
  'hg38_genes_intergenic',
  'hg38_genes_intronexonboundaries',
  'hg38_cpg_islands',
  'hg38_cpg_shores',
  'hg38_cpg_shelves',
  'hg38_cpg_inter')
dm_vs_kg_annotations = plot_annotation(
  annotated_regions = annotated,
  annotation_order = annots_order,
  plot_title = 'Annotations',
  x_label = 'knownGene Annotations',
  y_label = 'Count')
ggsave("plot2.png",dm_vs_kg_annotations , width = 6, height = 4, dpi = 300)

