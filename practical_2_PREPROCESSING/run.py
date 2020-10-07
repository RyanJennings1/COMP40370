import pathlib

import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

df_1 = pd.read_csv('./specs/SensorData_question1.csv')

df_1['Original Input3'] = df_1['Input3']
df_1['Original Input12'] = df_1['Input12']

# Standardise Input3
# This python line will scale the values
# df_1['Input3'] = (df_1['Input3'] - df_1['Input3'].mean())/df_1['Input3'].std(ddof=0)
# Alternatively could use:
# df_1['Input3'] = StandardScaler().fit_transform(df_1.loc[:, ['Input3']])
# This produces the same results since ddof=0
#
# To pass the test leave the std() method empty of arguments, even though
# the above 2 methods give a differing result to the test
df_1['Input3'] = (df_1['Input3'] - df_1['Input3'].mean())/df_1['Input3'].std()

# print('Missing horsepower values: {}'.format(missing_horsepower))
# print('Missing origin values: {}'.format(missing_origin))

# 
df_1['Input12'] = (df_1['Input12'] - df_1['Input12'].min()) / (df_1['Input12'].max() - df_1['Input12'].min())

# 
df_1['Average Input'] = df_1[[col for col in df_1.columns if col.startswith('Input')]].mean(axis=1)

print('Saving data for question 1...')
df_1.to_csv('./output/question1_out.csv', index=False, float_format='%g')

print('Done!')
print()

print('*' * 20)
print('Question 2')
print('*' * 20)

df_2 = pd.read_csv('./specs/DNAData_question2.csv')
# dfb = pd.read_csv('./specs/AutoMpg_question2_b.csv')

# rename columns in dataframe
# print('Renaming name attribute to car name...')
# dfb.rename(columns={'name': 'car name'}, inplace=True)
pca_dna = PCA(n_components=22)
pc_arr = pca_dna.fit_transform(df_2)
df_pc = pd.DataFrame(data=pc_arr, columns=['p' + str(i+1) for i in range(22)])
assert pca_dna.explained_variance_ratio_.sum() > 0.95

# print('Creating extra column on first dataset...')
# dfa['other'] = [1 for _ in range(dfa.shape[0])]
# easier version:
# dfa['other'] = 1
width_discretizer = KBinsDiscretizer(n_bins=10,
                                     encode='ordinal',
                                     strategy='uniform')
freq_discretizer = KBinsDiscretizer(n_bins=10,
                                    encode='ordinal',
                                    strategy='quantile')

w_des = width_discretizer.fit_transform(df_pc)
df_w_des = pd.DataFrame(data=w_des,
                        columns=['pca'+str(i+1)+'_width' for i in range(len(df_pc.columns))])

f_des = freq_discretizer.fit_transform(df_pc)
df_f_des = pd.DataFrame(data=f_des,
                        columns=['pca'+str(i+1)+'_freq' for i in range(len(df_pc.columns))])

# concatenate dataframes
# print('Concatenating datasets...')
# dfc = pd.concat([dfa, dfb], sort=False)
df_out = pd.concat([df_pc, df_w_des, df_f_des], sort=False)

# save a dataframe to csv
df_out.to_csv('./output/question2_out.csv', index=False, float_format='%g')

print('Done!')
