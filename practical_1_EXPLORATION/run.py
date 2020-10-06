import pathlib

import pandas as pd


# create the output directory if it does not exist
#
# documentation here:
# https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

df_1 = pd.read_csv('./specs/AutoMpg_question1.csv')

# check for missing values
#
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isnull.html
missing_horsepower = df_1['horsepower'].isnull().sum()
missing_origin = df_1['origin'].isnull().sum()

print('Missing horsepower values: {}'.format(missing_horsepower))
print('Missing origin values: {}'.format(missing_origin))

horsepower_mean = df_1['horsepower'].mean()

# replace missing values
#
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
print('Replacing missing values for horsepower...')
df_1['horsepower'].fillna(df_1['horsepower'].mean(), inplace=True)

print('Replacing missing values for origin...')
df_1['origin'].fillna(df_1['origin'].min(), inplace=True)

print('Saving data for question 1...')
df_1.to_csv('./output/question1_out.csv', index=False)

print('Done!')
print()

print('*' * 20)
print('Question 2')
print('*' * 20)

dfa = pd.read_csv('./specs/AutoMpg_question2_a.csv')
dfb = pd.read_csv('./specs/AutoMpg_question2_b.csv')

# rename columns in dataframe
#
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
print('Renaming name attribute to car name...')
dfb.rename(columns={'name': 'car name'}, inplace=True)

print('Creating extra column on first dataset...')
dfa['other'] = [1 for _ in range(dfa.shape[0])]
# easier version:
# dfa['other'] = 1

# concatenate dataframes
#
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
print('Concatenating datasets...')
dfc = pd.concat([dfa, dfb], sort=False)

# save a dataframe to csv, there are many options to do that
#
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
dfc.to_csv('./output/question2_out.csv', index=False, float_format='%g)

print('Done!')
