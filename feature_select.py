from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from feature_engine import discretisers as dsc
import pandas as pd
import numpy as np
from scipy.stats import entropy


def _entropy(x, base=2):
    value, counts = np.unique(x, return_counts=True, axis=0)  # If axis is not specified, input array x will be
    # flattened which is incorrect for computing H(x,y) where each element is a now a tuple, hence we define axis=0
    return entropy(counts, base=base)


def _compute_symmetric_uncertainty(fx, fy):
    """
    This function calculates the symmetrical uncertainty, where su(fx,fy) = 2*IG(fx,fy)/(H(fx)+H(fy))
    :param fx: {numpy array}, shape (n_samples,)
    :param fy: {numpy array}, shape (n_samples,)
    :return: su_score is the symmetrical uncertainty between fx and fy
    """
    hx = _entropy(fx)
    hy = _entropy(fy)

    # mutual_info_classif requires 2d matrix as input, thus we reshape fx to (n_samples,1)
    # mutual_info_classif computes information gain in log_e. To make it in log_2, we divide by np.log(2)
    # ig = mutual_info_classif(fx.reshape(-1, 1), fy)[0]/np.log(2)

    ig = hx + hy - _entropy(list(zip(fx, fy)))
    su_score = 2.0 * ig / (hx + hy)
    return su_score


def getFirstElement(s_list):
    """
    Return fp (first element in sorted s_list)
    :param s_list:
    :return: (feature, SU_Score, feature_index)
    """
    t = np.where(s_list[:, 2] > 1.)[0]
    if len(t):
        return int(s_list[t[0], 0]), s_list[t[0], 1], t[0]
    return None, None, None


def getNextElement(s_list, fp_idx):
    """
    Return the next element fq for fp
    :param s_list:
    :param fp_idx:
    :return: (feature, SU_Score, feature_index)
    """
    t = np.where(s_list[:, 2] > 1.)[0]
    t = t[t > fp_idx]
    if len(t):
        return int(s_list[t[0], 0]), s_list[t[0], 1], t[0]
    return None, None, None


def compute_fcbf(X, y, delta=0.0, redundancy_margin=0.1):
    """
    :param redundancy_margin:
    :param X: Feature Matrix
    :param y: target label vector
    :param delta: threshold for selecting 'relevant' features.
    :return: FCBF from [1]

    Reference
    ---------
    [1] Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data:
    A Fast Correlation-Based Filter Solution." ICML 2003.
    """
    n_samples, n_features = X.shape
    s_list = np.zeros((n_features, 3))
    s_list[:, -1] = 2.  # Initialise the non-redundant features
    # correlation_matrix = np.zeros((n_features, n_features + 1), dtype=float)
    # np.fill_diagonal(correlation_matrix, 1.0, wrap=True)
    # confidence_in_drop = []

    # Part-1: Identify relevant features
    for i in range(n_features):
        s_list[i, 0] = i
        s_list[i, 1] = _compute_symmetric_uncertainty(X[:, i], y)
        # correlation_matrix[i, -1] = round(s_list[i, 1], 4)
    s_list = s_list[s_list[:, 1] >= delta, :]
    s_list = s_list[np.argsort(s_list[:, 1])[::-1]]  # Sort the s_list in decreasing order of their SU values

    # Part-2: Remove redundant features from relevant ones
    fpq_su = {}
    fp, fp_su, fp_idx = getFirstElement(s_list)
    while fp is not None:
        fq, fq_su, fq_idx = getNextElement(s_list, fp_idx)
        if fq is not None:
            while fq is not None:
                if (fp, fq) not in fpq_su:
                    fpq_su[(fp, fq)] = _compute_symmetric_uncertainty(X[:, fp], X[:, fq])
                if fpq_su[(fp, fq)] >= fq_su:
                    # confidence_in_drop.append(fpq_su[(fp, fq)]-fq_su)
                    if fpq_su[(fp, fq)] - fq_su >= redundancy_margin:
                        s_list[fq_idx, -1] = 0.  # Remove fq with high confidence
                    else:
                        s_list[fq_idx, -1] = 1.  # Remove fq with low confidence
                # correlation_matrix[fp, fq] = round(fpq_su[(fp, fq)], 4)
                # correlation_matrix[fq, fp] = round(fpq_su[(fp, fq)], 4)
                fq, fq_su, fq_idx = getNextElement(s_list, fq_idx)
        fp, fp_su, fp_idx = getNextElement(s_list, fp_idx)

    # # Adding feature indices to correlation matrix
    # column_to_be_added = np.array([i for i in range(n_features)])
    # row_to_be_added = np.array([-1] + [i for i in range(n_features + 1)])
    # correlation_matrix = np.column_stack((column_to_be_added, correlation_matrix))
    # correlation_matrix = np.row_stack((row_to_be_added, correlation_matrix))

    s_best = s_list[s_list[:, 2] == 2., :2]  # 2: Relevant Features
    s_high_conf_dropped = s_list[s_list[:, 2] == 0., :2]  # 0: High Confidence Redundant Features
    s_low_conf_dropped = s_list[s_list[:, 2] == 1., :2]  # 1: Low Confidence Redundant Features

    return np.array(s_best[:, 0], dtype=int), np.array(s_best[:, 1]), \
           np.array(s_high_conf_dropped[:, 0], dtype=int), np.array(s_high_conf_dropped[:, 1]), \
           np.array(s_low_conf_dropped[:, 0], dtype=int), np.array(s_low_conf_dropped[:, 1])