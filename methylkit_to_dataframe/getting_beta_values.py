
import pandas as pd 

#reading the merged csv containing the number of Cs and number of Ts per sample 
df=pd.read_csv('meth.csv')

sample_nums = range(1, 72) 

#iteration
for sample_num in sample_nums:
    #calculate the column indices for Cs and Ts for the current sample
    cs_col = f'numCs{sample_num}' #fstring for simplicity
    ts_col = f'numTs{sample_num}'
    
    #calculate the beta value for number of methylated Cs 
    df[f'sample{sample_num}'] = (df[cs_col] / (df[cs_col] + df[ts_col]))


#list needs to be in correct order 
srr_numbers = ["SRRxxx",...,"SRRxx"]  #list of SRR numbers 

#zip to have sample to corresponding SRR number 
for sample_num, srr_num in zip(sample_nums, srr_numbers):
    #rename the column from sample{sample_num} to the corresponding SRR number
    df = df.rename(columns={f'sample{sample_num}': srr_num})

columns_to_drop = df.filter(regex='^(coverage|num)').columns
df.drop(columns=columns_to_drop, inplace=True)

df.to_csv('samples_beta_value.csv')
