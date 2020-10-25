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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

# Load the first csv file of marks data
df_1 = pd.read_csv('./specs/marks_question1.csv')

print('Plotting midterm exam scores vs final exam scores ...')
plt.scatter(df_1['midterm'], df_1['final'])
ax = plt.gca()
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xlabel('Final Score')
ax.set_ylabel('Midterm Score')
ax.set_title('Plot of Midterm Exam Scores vs Final Exam Scores')
plt.show()
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
y_vals = lin_reg.intercept_ + lin_reg.coef_[0] * x_vals
plt.plot(x_vals, y_vals, '--')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xlabel('Final Score')
ax.set_ylabel('Midterm Score')
ax.set_title('Plot of Midterm Exam Scores vs Final Exam Scores')
plt.show()
plt.tight_layout()
print('Writing plot to ./output/marks_line_best_fit.png ...')
plt.savefig('output/marks_line_best_fit.png', dpi=199)

print('Predicting the final mark for a midterm of 86 for question 1.3 ...')
predicted_value = lin_reg.predict(np.array([[86]]))
print(f'For a midterm exam score the final exam score is predicted to be: {predicted_value}')

print('Done!')
print()

print('*' * 20)
print('Question 2')
print('*' * 20)

# Load the second csv file of borrower data
df_2 = pd.read_csv('./specs/borrower_question2.csv')

# Filter out TID attribute from dataset
print('Dropping attribute "TID"...')
del df_2['TID']

print('Formatting our input data for Question 2 to use with our models ...')
# Change HomeOwner, MaritalStatus and DefaultedBorrower fields from strings to floats
df_2['HomeOwner'] = df_2['HomeOwner'].apply(lambda x: 1.0 if x == 'Yes' else 0.0)
df_2['MaritalStatus'] = df_2['MaritalStatus'].map({
    'Single': 0.0,
    'Married': 1.0,
    'Divorced': 2.0
    })
df_2['DefaultedBorrower'] = df_2['DefaultedBorrower'].apply(lambda x: 1.0 if x == 'Yes' else 0.0)

print('Splitting training and test data ...')
# Thankfully here we are provided with a field that tells us if
# a borrower defaulted on their loan to see how the other fields
# impacted the result
training_data = df_2.drop('DefaultedBorrower', axis=1)
test_data = df_2['DefaultedBorrower']

print('Creating high decision tree classifier ...')
clf_high = DecisionTreeClassifier(criterion='entropy',
                                  min_impurity_decrease=0.5)
clf_high_model = clf_high.fit(training_data, test_data)

plt.figure()
plot_tree(clf_high_model)
plt.show()
plt.savefig('./output/tree_high.png')

print('Creating low threshold decision tree classifier ...')
clf_low = DecisionTreeClassifier(criterion='entropy',
                                 min_impurity_decrease=0.1)
clf_low_model = clf_low.fit(training_data, test_data)

# Set the figsize so we can clearly see the arrows between
# each leaf in the decision tree
plt.figure(figsize=(18, 8))
plot_tree(clf_low_model, filled=True)
plt.show()
plt.tight_layout()
plt.savefig('./output/tree_low.png')

print('Done!')
