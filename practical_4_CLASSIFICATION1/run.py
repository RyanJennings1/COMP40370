"""
  name: run.py
  author: Ryan Jennings
  date: 2020-10-22
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# Create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

# Load the first csv file of sensor data
df_1 = pd.read_csv('./specs/marks_question1.csv')

print('Plotting midterm exam scores vs final exam scores ...')
plt.scatter(df_1['midterm'], df_1['final'])
ax = plt.gca()
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xlabel('Final Score')
ax.set_ylabel('Midterm Score')
ax.set_title('Plot of Midterm Exam Scores vs Final Exam Scores')
plt.tight_layout()
print('Writing plot to ./output/marks.png ...')
plt.savefig('output/marks.png', dpi=199)

print('Generating linear regression model from data for question 1.2 ...')
lin_reg = LinearRegression().fit(df_1['midterm'].values.reshape(-1, 1),
                                 df_1['final'])

print(f'Linear Regression model coeffecient: {lin_reg.coef_}')
print(f'Linear Regression model intercept: {lin_reg.intercept_}')

print('Plotting line of best fit with scatter plot to help describe model ...')
plt.scatter(df_1['midterm'], df_1['final'])
ax = plt.gca()
x_vals = np.linspace(0, 100)
y_vals = lin_reg.intercept_ + lin_reg.coef[0] * x_vals
plt.plot(x_vals, y_vals, '--')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xlabel('Final Score')
ax.set_ylabel('Midterm Score')
ax.set_title('Plot of Midterm Exam Scores vs Final Exam Scores')
plt.tight_layout()
print('Writing plot to ./output/marks_line_best_fit.png ...')
plt.savefig('output/marks_line_best_fit.png', dpi=199)

print('Predicting the final mark for a midterm of 86 for question 1.3 ...')
predicted_value = lin_reg.predict(np.array([[86]])
print(f'For a midterm exam score the final exam score is predicted to be: {predicted_value}')

print('Done!')
print()

print('*' * 20)
print('Question 2')
print('*' * 20)

# Load the second csv file of DNA data
df_2 = pd.read_csv('./specs/borrower_question2.csv')

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
