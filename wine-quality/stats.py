import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Plot feature importances
def plot_feature_importances(model):
    n_features = data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), wine.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


wine = pd.read_csv("data/winequality-red.csv", sep=';')

# print(wine.shape)
# print(wine.keys())
# print(wine.iloc[:, 0:11].shape)
# print(wine.iloc[:, 11])

# TODO: Remove outliers
# print(wine.describe())

# Put data into correct format. X data will be all columns except quality. y data will be quality (target)
data = wine.iloc[:, 0:11]
target = wine.iloc[:, 11]

sulphates = wine.iloc[:, 9]
alcohol = wine.iloc[:, 10]

# print(data)
# print(target)

# Pearson's Correlation Coefficient between alcohol and quality = (0.47616632400113607, 2.8314769747769775e-91)
# p-value is number between 0 and 1 represeting probability data  2.8314769747769775e-91
# r = 0.47616632400113607

# Stats information excluding quality
feature_names = wine.columns.values[:11]
# New dict with column names
stats = {k: {} for k in feature_names}

# Print pearson for each column
for i in range(11):
    feature = wine.iloc[:, i]
    feature_name = feature.name
    pearson_r = sp.stats.pearsonr(feature, target)

    # add pearson's r information to dict
    stats[feature_name]['pearson_r'] = pearson_r[0]
    print('', feature_name, 'r: ', pearson_r[0])

print(stats)

# TODO: create horizontal bar chart and compare to feature importances

# pearson_alcohol = sp.stats.pearsonr(alcohol, target)
# pearson_sulphates = sp.stats.pearsonr(sulphates, target)
#
# print(pearson_alcohol)
# print(pearson_sulphates)

# Linear regression with 1 independent variable

# Split training and test set
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

# Ridge Regression
# ridge = Ridge().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# Random Forest Regression
# forest = RandomForestRegressor(n_estimators=1000, max_features=3, random_state=2)
# forest.fit(X_train, y_train)
# print("Training set score: {:.2f}".format(forest.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(forest.score(X_test, y_test)))
# plot_feature_importances(forest)

# grr = pd.plotting.scatter_matrix(wine, c='y', figsize=(20, 20), marker='o',
#                         hist_kwds={'bins': 20}, s=60, alpha=.8)
# plt.show()

# forest = GradientBoostingRegressor(n_estimators=1000, max_features=3, max_depth=50, learning_rate=0.0017, random_state=2)
# forest.fit(X_train, y_train)
# print("Training set score: {:.2f}".format(forest.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(forest.score(X_test, y_test)))
# plot_feature_importances(forest)

# Useful methods
# df.to_excel()