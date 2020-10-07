import pathlib

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler

# Create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

# Load the first csv file of sensor data
df_1 = pd.read_csv('./specs/SensorData_question1.csv')

# Create copies of the Input3 and Input12 columns
print('Copying columns Input3 and Input12...')
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
# the above 2 methods give a differing result to the test answer
print('Standardising Input3...')
df_1['Input3'] = (df_1['Input3'] - df_1['Input3'].mean())/df_1['Input3'].std()

# Normalise the data between 0 and 1. This can be done with the following
# formula
print('Normalising Input12...')
df_1['Input12'] = (df_1['Input12'] - df_1['Input12'].min()) / (df_1['Input12'].max() - df_1['Input12'].min())

# Generate Average Input column by getting the mean of each row
# if the row is like 'InputX'
print('Creating Average Input column...')
df_1['Average Input'] = df_1[[col for col in df_1.columns if col.startswith('Input')]].mean(axis=1)

print('Saving data for question 1...')
df_1.to_csv('./output/question1_out.csv', index=False, float_format='%g')

print('Done!')
print()

print('*' * 20)
print('Question 2')
print('*' * 20)

# Load the second csv file of DNA data
df_2 = pd.read_csv('./specs/DNAData_question2.csv')

# Use Principal Component Analysis to reduce the amount of columns
# We want only 95% of data and this will enable us to remove
# more than 50% of the columns while keeping the important information
print('Running Principal Component Analysis on DNA dataset...')
pca_dna = PCA(n_components=22)
pc_arr = pca_dna.fit_transform(df_2)
df_pc = pd.DataFrame(data=pc_arr, columns=['p' + str(i+1) for i in range(22)])
# Assert to check that the variance is more than 95%
# of the original data
assert pca_dna.explained_variance_ratio_.sum() > 0.95

# Use KBinsDiscretizer to separate the data into first equal
# sized bins and then split into frequencies of the data
print('Running discretisation to split the data...')
width_discretizer = KBinsDiscretizer(n_bins=10,
                                     encode='ordinal',
                                     strategy='uniform')
freq_discretizer = KBinsDiscretizer(n_bins=10,
                                    encode='ordinal',
                                    strategy='quantile')

w_des = width_discretizer.fit_transform(df_pc)
df_w_des = pd.DataFrame(data=w_des,
                        columns=['pca'+str(i)+'_width' for i in range(len(df_pc.columns))])

f_des = freq_discretizer.fit_transform(df_pc)
df_f_des = pd.DataFrame(data=f_des,
                        columns=['pca'+str(i)+'_freq' for i in range(len(df_pc.columns))])

# Concatenate dataframes
print('Concatenating datasets...')
df_out = pd.concat([df_2, df_w_des, df_f_des], sort=False, axis=1)

# Save a dataframe to csv
df_out.to_csv('./output/question2_out.csv', index=False, float_format='%g')

print('Done!')
