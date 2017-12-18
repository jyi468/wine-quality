import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

wine = pd.read_csv("data/winequality-red.csv", sep=';')


def plot_pearson_r_horizontal(winedata, stat):
    # Print pearson for each column
    for i in range(11):
        feature = winedata.iloc[:, i]
        feature_name = feature.name
        pearson_r = sp.stats.pearsonr(feature, target)

        # add pearson's r information to dict
        stat[feature_name]['pearson_r'] = pearson_r[0]
        # add to pearson_list with absolute values to compare with feature importances
        pearson_list.append(math.fabs(pearson_r[0]))
        # print('', feature_name, 'r: ', pearson_r[0])

    # y label are features, labels going in reverse order
    # x should be pearson r value
    y_pos = np.arange(11)

    fig, ax = plt.subplots()

    ax.barh(y_pos, pearson_list)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Pearson R (abs)')

    plt.show()


# Plot histograms for all features
def plot_feature_histograms(winedata):
    # Create subplot matrix of 3 rows and 4 columns
    fig, axes = plt.subplots(3, 4)
    axes = axes.ravel()
    # Print pearson for each column
    for i, ax in enumerate(axes):
        feature = winedata.iloc[:, i]
        feature_name = feature.name
        # num_bins must be appropriate for data spread - use descriptive stats to determine?
        if i == 11:
            num_bins = 5
        else:
            num_bins = 40
        ax.hist(feature, num_bins, alpha=0.5, histtype='bar', edgecolor='black')
        ax.set_title(feature_name)
        ax.set_ylabel('Count')

    # plt.tight_layout(pad=0.25, rect=[0, 0, 1, 1])
    fig.suptitle('Feature Counts', fontsize=16)
    plt.subplots_adjust(top=0.9,
                        bottom=0.042,
                        left=0.037,
                        right=0.99,
                        hspace=0.241,
                        wspace=0.205)
    plt.show()


# Plot feature importances
def plot_feature_importances(model):
    n_features = data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), wine.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


# print(wine.shape)
# print(wine.keys())
# print(wine.iloc[:, 0:11].shape)
# print(wine.iloc[:, 11])

# TODO: Remove outliers
print(wine.describe())


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
pearson_list = []

# Create histograms of data distributions for all features - show in one plot
# Create box and whisker plots
# Show data fitting before and after outliers
# Show most important positive and negative features

# plot_pearson_r_horizontal(wine, stats)
plot_feature_histograms(wine)

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
