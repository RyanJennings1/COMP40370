"""
  name: run.py
  author: Ryan Jennings
  date: 2020-10-11
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# Create the output directory if it does not exist
pathlib.Path('./output').mkdir(exist_ok=True)

print('*' * 20)
print('Question 1')
print('*' * 20)

# Load the first csv file of x, y data
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

# Load the second csv file of cereal data
df_2 = pd.read_csv('./specs/question_2.csv')

# Filter out attributes we won't be using from dataset
print('Dropping attributes NAME, MANUF, TYPE and RATING ...')
df_2 = df_2.drop(['NAME', 'MANUF', 'TYPE', 'RATING'], axis=1)

print('Generating different KMeans clusterings and adding to dataframe ...')
df_2_kmeans1 = KMeans(n_clusters=5,
                      random_state=0,
                      n_init=5,
                      max_iter=100).fit(df_2)
df_2_kmeans2 = KMeans(n_clusters=5,
                      random_state=0,
                      n_init=100,
                      max_iter=100).fit(df_2)
df_2_kmeans3 = KMeans(n_clusters=3,
                      random_state=0,
                      n_init=100,
                      max_iter=100).fit(df_2)

df_2['config1'] = df_2_kmeans1.labels_
df_2['config2'] = df_2_kmeans2.labels_
df_2['config3'] = df_2_kmeans3.labels_

print('Saving data for question 2.7 ...')
df_2.to_csv('./output/question_2.csv', index=False)

print('Done!')
print()

print('*' * 20)
print('Question 3')
print('*' * 20)

# Load the third file of 2-dimensional points data
df_3 = pd.read_csv('./specs/question_3.csv')
print('Discarding ID attribute ...')
df_3 = df_3.drop(['ID'], axis=1)

print('Generating KMeans for question 3.1 ...')
df_3_kmeans1 = KMeans(n_clusters=7,
                      n_init=5,
                      max_iter=100,
                      random_state=0).fit(df_3)
df_3['kmeans'] = df_3_kmeans1.labels_

plt.scatter(df_3['x'], df_3['y'], c=df_3['kmeans'], cmap='brg')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plot of x, y Coloured by KMeans Clustering')
plt.tight_layout()
print('Writing plot to ./output/question_3_1.pdf ...')
plt.savefig('output/question_3_1.pdf', dpi=199)
plt.show()

print('Normalising question_3 x and y coordinates ...')
df_3_norm = MinMaxScaler().fit_transform(df_3[['x', 'y']])

print('Using DBSCAN to cluster points ...')
df_3_dbscan1 = DBSCAN(eps=0.04, min_samples=4).fit(df_3_norm)
df_3['dbscan1'] = df_3_dbscan1.labels_

plt.scatter(df_3['x'], df_3['y'], c=df_3['dbscan1'], cmap='brg')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plot of x, y Coloured by DBSCAN Clustering epsilon=0.04')
plt.tight_layout()
print('Writing plot to ./output/question_3_2.pdf ...')
plt.savefig('output/question_3_2.pdf', dpi=199)
plt.show()

df_3_dbscan2 = DBSCAN(eps=0.08, min_samples=4).fit(df_3_norm)
df_3['dbscan2'] = df_3_dbscan2.labels_

plt.scatter(df_3['x'], df_3['y'], c=df_3['dbscan2'], cmap='brg')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plot of x, y Coloured by DBSCAN Clustering epsilon=0.08')
plt.tight_layout()
print('Writing plot to ./output/question_3_3.pdf ...')
plt.savefig('output/question_3_3.pdf', dpi=199)
plt.show()

print('Saving data for question 3.5 ...')
df_3.to_csv('./output/question_3.csv', index=False)

print('Done!')
