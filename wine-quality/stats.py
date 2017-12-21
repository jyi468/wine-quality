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


# Pearson's Correlation Coefficient between alcohol and quality = (0.47616632400113607, 2.8314769747769775e-91)
# p-value is number between 0 and 1 represeting probability data  2.8314769747769775e-91
# r = 0.47616632400113607
def plot_pearson_r_horizontal(winedata, stat, target, feature_names, abs=False):
    pearson_list = []
    # pearson_alcohol = sp.stats.pearsonr(alcohol, target)
    # pearson_sulphates = sp.stats.pearsonr(sulphates, target)
    #
    # print(pearson_alcohol)
    # print(pearson_sulphates)

    # Print pearson for each column
    for i in range(11):
        feature = winedata.iloc[:, i]
        feature_name = feature.name
        pearson_r = sp.stats.pearsonr(feature, target)

        # add pearson's r information to dict
        stat[feature_name]['pearson_r'] = pearson_r[0]
        # print('', feature_name, 'r: ', pearson_r[0])
        # add to pearson_list with absolute values to compare with feature importances
        if abs:
            pearson_list.append(math.fabs(pearson_r[0]))
            title = 'Pearson R (abs)'
        else:
            pearson_list.append(pearson_r[0])
            title = 'Pearson R'

    # y label are features, labels going in reverse order
    # x should be pearson r value
    y_pos = np.arange(11)

    fig, ax = plt.subplots()

    ax.barh(y_pos, pearson_list)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel(title)

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


# Plot box plots for all features
def plot_box(winedata):
    # Create subplot matrix of 3 rows and 4 columns
    fig, axes = plt.subplots(3, 4)
    axes = axes.ravel()
    # Print pearson for each column
    for i, ax in enumerate(axes):
        feature = winedata.iloc[:, i]
        feature_name = feature.name
        ax.boxplot(feature)
        ax.set_title(feature_name)
        ax.set_ylabel('Value')

    # plt.tight_layout(pad=0.25, rect=[0, 0, 1, 1])
    fig.suptitle('Box Plots', fontsize=16)
    plt.subplots_adjust(top=0.9,
                        bottom=0.042,
                        left=0.037,
                        right=0.99,
                        hspace=0.241,
                        wspace=0.205)
    plt.show()


# Plot feature importances
def plot_feature_importances(data, model):
    n_features = data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), wine.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


def plot_ml(data, target, type):
    # Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

    if type == "ridge":
        # Ridge Regression
        ridge = Ridge().fit(X_train, y_train)
        print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
        print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
    elif type == "rfr":
        # Random Forest Regression
        forest = RandomForestRegressor(n_estimators=1000, max_features=3, random_state=2)
        forest.fit(X_train, y_train)
        print("Training set score: {:.2f}".format(forest.score(X_train, y_train)))
        print("Test set score: {:.2f}".format(forest.score(X_test, y_test)))
        plot_feature_importances(data, forest)
        plt.show()
    elif type == "gbr":
        forest = GradientBoostingRegressor(n_estimators=1000, max_features=3, max_depth=50, learning_rate=0.0017,
                                           random_state=2)
        forest.fit(X_train, y_train)
        print("Training set score: {:.2f}".format(forest.score(X_train, y_train)))
        print("Test set score: {:.2f}".format(forest.score(X_test, y_test)))
        plot_feature_importances(data, forest)


def scatter_wine(data):
    # Most recent pandas uses pd.plotting.scatter_matrix
    grr = pd.scatter_matrix(data, figsize=(20, 20), marker='o',
                            hist_kwds={'bins': 20}, s=60, alpha=.8)
    plt.show()


def main():
    # TODO: Remove outliers
    # print(wine.describe())

    # Put data into correct format. X data will be all columns except quality. y data will be quality (target)
    data = wine.iloc[:, 0:11]
    target = wine.iloc[:, 11]

    sulphates = wine.iloc[:, 9]
    alcohol = wine.iloc[:, 10]

    # Stats information excluding quality
    feature_names = wine.columns.values[:11]
    # New dict with column names
    stats = {k: {} for k in feature_names}

    # Main Plotting/Analysis Functionality
    # plot_pearson_r_horizontal(wine, stats, target, feature_names)
    plot_box(wine)
    # plot_feature_histograms(wine)
    # plot_ml(data, target, "rfr")
    # scatter_wine(wine)

    # TODO list
    # Show data fitting before and after outliers
    # Show most important positive and negative features
    # See what happens after a treatment
    # ANOVA and other tests

main()

# Useful methods
# df.to_excel()
