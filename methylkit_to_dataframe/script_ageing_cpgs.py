#Script which only keeps the CpGs associated with teh genes in OpenGenes and Epigenetic clocks 
#Importing the right packages 
import pandas as pd

#final dataframe which will be converted into a csv file
df_final=pd.DataFrame(columns=["seqnames","start","end","width","strand","annot.seqnames","annot.start","annot.end","annot.width","annot.strand","annot.id","annot.tx_id","annot.gene_id","annot.symbol","annot.type"])

#importing the csv file containing all the ageing genes
df_genes=pd.read_csv('all_genes.csv')

#Importing the csv file containing the annotations and the different chromosomal locations
df=pd.read_csv('annotated_RRBS_sample.csv') 

#Creating a list from those genes 
genes_list=list(df_genes['GENES'])

#writing file 
with open("cpgs_ageing_complete_2.csv",'w') as file:
    file.write(','.join(df_final.columns.astype(str))+'\n')
    for index, row in df.iterrows(): 
        if row['annot.symbol'] in genes_list: 
            file.write(','.join(row.astype(str)) + '\n')
