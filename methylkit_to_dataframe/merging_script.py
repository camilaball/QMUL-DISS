#import right package 
import pandas as pd 

#load the csv files 
df1=pd.read_csv('cpgs_genes_final_2.csv')
df2=pd.read_csv('samples_beta_value.csv')

#drop useless columns 
df2=df2.drop(columns=['Unnamed: 0'])

#merge the dataframes on chromosome number, start, end positions 
merged_df=pd.merge(df1,df2, left_on=['seqnames','start','end'], right_on=['chr','start','end'])

#final file doesn't need all columns so need to select the right columns 
final_df=merged_df[['seqnames','start','end','annot.symbol']+list(df2.columns[5:])]

#Renaming 
final_df.columns=['chr','start','end','gene_symbol']+list(final_df.columns[4:])

#saving
final_df.to_csv('merged_genes_beta_2.csv',index=False)
