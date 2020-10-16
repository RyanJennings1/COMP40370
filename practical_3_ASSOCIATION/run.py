"""
  name: run.py
  author: Ryan Jennings
  date: 2020-10-13
"""
import pathlib

import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

from mlxtend.preprocessing import TransactionEncoder

# Create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

# Load the first csv file of sensor data
df_1 = pd.read_csv('./specs/gpa_question1.csv')

# Filter out count attribute from dataset
print('Dropping attribute "count"...')
del df_1['count']

# Create a method that can be reused for question 2
def format_frequent_itemsets(dataframe: pd.DataFrame) -> pd.DataFrame:
    # The TransactionEncoder can not take a dataframe as an input
    # so we simply need to convert the dataframe to an array of
    # arrays, each containing a row of the dataset
    # This for loop modified from tutorial example here:
    # https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
    records = []
    for i in range(0, len(dataframe)):
        records.append([str(dataframe.values[i, j]) for j in range(0, len(dataframe.columns))])

    # From mlxtend's documentation on the Apriori algorithm
    # http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    print('Encoding the dataset for the Apriori algorithm...')
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    return pd.DataFrame(te_ary, columns=te.columns_)

df_ap = format_frequent_itemsets(dataframe=df_1)

# Call the apriori algorithm on our data with a min support
# of 15%
ap = apriori(df_ap, min_support=0.15, use_colnames=True)

print('Saving data for question 1.2...')
ap.to_csv('./output/question1_out_apriori.csv', index=False, float_format='%g')

# Use the association_rules method from the mlxtend libray
# to generate a dataframe of frequent itemsets with more than
# 90% confidence
assoc_rules_90_pc = association_rules(ap,
                                      metric="confidence",
                                      min_threshold=0.9)
print('Saving data for question 1.4...')
assoc_rules_90_pc.to_csv('./output/question1_out_rules9.csv', index=False, float_format='%g')

# Similar to the last question but for 70% confidence
assoc_rules_70_pc = association_rules(ap,
                                      metric="confidence",
                                      min_threshold=0.7)
print('Saving data for question 1.6...')
assoc_rules_70_pc.to_csv('./output/question1_out_rules7.csv', index=False, float_format='%g')

print('Done!')
print()

print('*' * 20)
print('Question 2')
print('*' * 20)

# Load the second csv file of DNA data
df_2 = pd.read_csv('./specs/bank_data_question2.csv')

# Filter out id attribute from dataset
print('Dropping attribute "id"...')
del df_2['id']

# Discretize the numeric columns (age, income, children) into 3 equal bins
# Replace the columns with their discretized equivalents
for num_key in ['age', 'income', 'children']:
    df_2[num_key] = pd.cut(df_2[num_key], bins=3)

# Yes/No values on their own result in illegible results for our output
# so modify the values to represent the answer plus their field name
for field in ['married', 'car', 'save_act', 'current_act', 'mortgage', 'pep']:
    df_2[field] = df_2[field].replace('YES', f'YES_{field}')
    df_2[field] = df_2[field].replace('NO', f'NO_{field}')

# Use the formatting method from earlier
df_fp = format_frequent_itemsets(dataframe=df_2)

# Use the FP-Growth algorithm to generate frequent itemsets
fp = fpgrowth(df_fp, min_support=0.2, use_colnames=True)

# Save a dataframe to csv
fp.to_csv('./output/question2_out_fpgrowth.csv', index=False, float_format='%g')

# Produce association rules. I found that a threshold of 0.79 after,
# repeated testing, produced exactly 10 rules.
# The lowest being 0.790576 so that would be the exact threshold value
# for 10 association rules
assoc_rules_79_pc = association_rules(fp,
                                      metric="confidence",
                                      min_threshold=0.79)

# Save a dataframe to csv
assoc_rules_79_pc.to_csv('./output/question2_out_rules.csv', index=False, float_format='%g')

print('Done!')
