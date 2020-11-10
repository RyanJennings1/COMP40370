"""
  name: run.py
  author: Ryan Jennings
  date: 2020-10-11
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.clustering import KMeans

# Create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

# Load the first csv file of marks data
df_1 = pd.read_csv('./specs/question_1.csv')

print('Performing KMeans clustering on question 1 dataset ...')
kmeans = KMeans(n_clusters=3, random_state=0)
df_1_kmeans = kmeans.fit(df_1)
df_1['cluster'] = df_1_kmeans.labels_

print('Saving data for question 1.2 ...')
df_1.to_csv('./output/question_1.csv', index=False)

print('Plotting scatter plot with points coloured by their clustering ...')
plt.scatter(df_1['x'], df_1['y'], c=df_1['cluster'], cmap='brg')
ax = plt.gca()
ax.set_xlim([0, 25])
ax.set_ylim([0, 25])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plot of x, y Coloured by Cluster')
plt.tight_layout()
print('Writing plot to ./output/question_1.pdf ...')
plt.savefig('output/question_1.pdf', dpi=199)
plt.show()

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
plt.savefig('./output/tree_high.png')
plt.show()

print('Creating low threshold decision tree classifier ...')
clf_low = DecisionTreeClassifier(criterion='entropy',
                                 min_impurity_decrease=0.1)
clf_low_model = clf_low.fit(training_data, test_data)

# Set the figsize so we can clearly see the arrows between
# each leaf in the decision tree
plt.figure(figsize=(18, 8))
plot_tree(clf_low_model, filled=True)
plt.tight_layout()
plt.savefig('./output/tree_low.png')
plt.show()

print('Done!')
