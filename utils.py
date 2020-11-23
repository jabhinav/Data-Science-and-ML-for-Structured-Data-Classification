import numpy as np
from matplotlib import pyplot
from models import get_classifiers
from sklearn.inspection import permutation_importance
from sklearn.ensemble import IsolationForest


def remover_outliers(X, y, data_preprocess_obj):
    # Outlier detect
    retained_index = data_preprocess_obj.processed_df.index.to_numpy()
    # clf = LocalOutlierFactor()
    clf = IsolationForest(n_estimators=400, random_state=2)
    outlier_list = clf.fit_predict(X)
    score = (outlier_list == 1).sum() / len(outlier_list)
    outlier_index_array = np.where(outlier_list == -1)
    outlier_index = (retained_index[outlier_index_array]).tolist()
    print("{} Outliers to be removed: {}".format(len(outlier_index), outlier_index_array))
    print("Number of positive class samples removed: {}".format(sum(y[outlier_index])))
    X = np.delete(X, outlier_index_array, axis=0)
    y = np.delete(y, outlier_index_array, axis=0)
    return X, y, score


def box_plot(results, names, stage, save_fig=False):
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.xticks(rotation=90)
    # Pad margins so that markers don't get clipped by the axes
    pyplot.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    pyplot.subplots_adjust(bottom=0.2)
    if save_fig:
        pyplot.savefig("./box_plot.png")
        pyplot.clf()
    else:
        pyplot.show()


def feature_importance_plot(model_name, X, y, labels, regression_based=False, tree_based=False,
                            no_intrinsic_selection=False, save_fig=False):
    # Feature Importance Visualisation
    model = get_classifiers()[model_name]
    model.fit(X, y)
    importance = []

    # For Regression based Models
    if regression_based:
        importance = model.coef_[0]

    # For Tree based models
    if tree_based:
        importance = model.feature_importances_

    # For models that do not support intrinsic feature importance scores
    if no_intrinsic_selection:
        results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
        importance = results.importances_mean

    pyplot.bar([x for x in range(len(importance))], sorted(importance))
    pyplot.xticks([x for x in range(len(importance))],
                  labels=[y for _, y in sorted(zip(importance, labels), key=lambda pair: pair[0])],
                  rotation=90)
    pyplot.margins(0.2)
    pyplot.subplots_adjust(bottom=0.2)
    if save_fig:
        pyplot.savefig("./{}_Feature Importance".format(model_name))
    pyplot.show()
    # pyplot.clf()


# scatter plot of dataset, different color for each class
def plot_dataset(X, y, feature_x, feature_y, ):
    # create scatter plot for samples from each class
    n_classes = len(np.unique(y))
    for class_value in range(n_classes):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)[0]
        # create scatter of these samples
        pyplot.scatter(X[row_ix, feature_x], X[row_ix, feature_y], label=str(class_value))
    # show a legend
    pyplot.legend()
    # show the plot
    pyplot.savefig("./vis_data_fx{}_{}".format(feature_x, feature_y))


def do_prelim_data_analysis(data_df):
    # Print information about categorical and numerical features
    print(data_df.info())
    print(data_df.describe(include='all'))

    # Data Analysis: Numerical Features
    # select columns with numerical data types
    num_ix = data_df.select_dtypes(include=['int64', 'float64']).columns
    # select a subset of the dataframe with the chosen columns
    subset = data_df[num_ix]
    # create a histogram plot of each numeric variable
    ax = subset.hist()
    # disable axis labels to avoid the clutter
    for axis in ax.flatten():
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    # save the plot
    pyplot.savefig("./numerical_features_distribution.png")
    pyplot.clf()