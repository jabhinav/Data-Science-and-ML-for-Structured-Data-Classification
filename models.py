from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, fbeta_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from imblearn.under_sampling import TomekLinks, InstanceHardnessThreshold, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from xgboost import XGBClassifier
import timeit
import random
import numpy as np
import json
import os
import time

seed = 0
random.seed(seed)
NUM_PARALLEL_JOBS = -1


def compute_fbeta(y_test, y_pred):
    # False Negatives are costly, therefore we want to priortise recall over precision. Choose beta > 1. Using F2.
    # (If FP have more cost, use F0.5(Beta=0.5<1) score)
    return fbeta_score(y_test, y_pred, beta=2)


def compile_overall_scores_from_grid_search(result_dict, nsplits):
    overall_score = {model_name: 0 for model_name in get_classifiers().keys()}
    for split in result_dict.keys():
        for model in result_dict[split]['Score'].keys():
            overall_score[model] += result_dict[split]['Score'][model]
    for model in overall_score.keys():
        overall_score[model] /= nsplits
    print("Printing Avg Model Score k-fold cross validation")
    print(overall_score, "\n")


def run_mlp(X, y, F, h=(50, 50), alpha=0.001, max_iter=1000):
    if len(F) > 0:
        scores = cross_val_score(MLPClassifier(hidden_layer_sizes=h, alpha=alpha, max_iter=max_iter),
                                 X[:, F[0:]], y, cv=5)
        print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))


def run_classifier(clf, param_grid, X_train, y_train, X_test, y_test, compute_accuracy=False):
    score = None
    if compute_accuracy:
        cv_model = GridSearchCV(estimator=clf,
                                param_grid=param_grid,
                                scoring='accuracy',
                                cv=5,
                                n_jobs=NUM_PARALLEL_JOBS,
                                verbose=0)
        best_model = cv_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return round(score, 3), best_model.best_params_, best_model
    else:
        # cost_fn = make_scorer(compute_cost, greater_is_better=False)

        cost_fn = make_scorer(compute_fbeta)
        cv_model = GridSearchCV(estimator=clf,
                                param_grid=param_grid,
                                scoring=cost_fn,
                                cv=5,
                                n_jobs=NUM_PARALLEL_JOBS,
                                verbose=0)
        # Balance the dataset
        X_res, y_res = get_samplers()['RENN'].fit_resample(X_train, y_train)
        best_model = cv_model.fit(X_res, y_res)
        y_pred = best_model.predict(X_test)
        score = compute_fbeta(y_test, y_pred)
        return score, best_model.best_params_, best_model


def run_classifiers(X_train, y_train, X_test, y_test):
    result = {}
    params = {}
    models = {}
    comp_time = {}

    # # Ada-boost
    start = timeit.default_timer()
    clf = AdaBoostClassifier(random_state=seed)
    param_grid = {'n_estimators': range(300, 1000, 300)}
    result['AdaBoost'], params['AdaBoost'], models['AdaBoost'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['AdaBoost'] = stop - start

    # # SVM-Linear
    start = timeit.default_timer()
    clf = SVC(kernel="linear", random_state=seed)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    result['Linear-SVM'], params['Linear-SVM'], models['Linear-SVM'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['Linear-SVM'] = stop - start

    # # SVM-RBF
    start = timeit.default_timer()
    clf = SVC(kernel="rbf", random_state=seed)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}
    result['RBF-SVM'], params['RBF-SVM'], models['RBF-SVM'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['RBF-SVM'] = stop - start

    # # Decision Tree
    start = timeit.default_timer()
    clf = DecisionTreeClassifier(random_state=seed)
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(5, 40, 5),
        "min_samples_leaf": range(1, 30, 2),
    }
    result['D-Tree'], params['D-Tree'], models['D-Tree'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['D-Tree'] = stop - start

    # # Random Forest
    start = timeit.default_timer()
    clf = RandomForestClassifier(random_state=seed)
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 20, 30],
        # "min_samples_leaf": range(1, 30, 2),
        "max_features": range(4, 16, 4)
    }
    result['R-Forest'], params['R-Forest'], models['R-Forest'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['R-Forest'] = stop - start

    # # QDA
    start = timeit.default_timer()
    clf = QuadraticDiscriminantAnalysis()
    param_grid = {'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}
    result['QDA'], params['QDA'], models['QDA'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['QDA'] = stop - start

    # # Multi-layer Perceptron Classifier
    start = timeit.default_timer()
    clf = MLPClassifier()
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (50, 100, 50), (100,), (100, 200, 100)],
        # 'hidden_layer_sizes': [(100,), (50, 50), (50, 100, 50)],
        # 'alpha': 10.0 ** -np.arange(1, 6, 2),
        'alpha': [0.001],
        'max_iter': [1000],
        # 'learning_rate': ['constant', 'adaptive']
    }
    result['MLP'], params['MLP'], models['MLP'] = run_classifier(
        clf, param_grid, X_train, y_train, X_test, y_test)
    stop = timeit.default_timer()
    comp_time['MLP'] = stop - start

    return result, params, models, comp_time


def run_classifiers_kfold_with_grid_search(X, y, result_dir=None, num_splits=5, keyword=''):
    overall_results = []
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), desc="Folds", total=num_splits):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        result, params, models, comp_time = run_classifiers(X_train, y_train, X_test, y_test)
        per_split_result = {
            'Score': result,
            'Best_Params': params,
            'Computation_time': comp_time
        }
        overall_results.append(per_split_result)

    result_dict = {}
    for split_id, split_result in enumerate(overall_results):
        result_dict['split_{}'.format(split_id)] = split_result

    if result_dir:
        with open(os.path.join(result_dir, f'{keyword}.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)

    compile_overall_scores_from_grid_search(result_dict, nsplits=num_splits)


def evaluate_model(X, y, model, num_splits=5):
    # cost_fn = make_scorer(compute_cost, greater_is_better=False)
    metric = make_scorer(compute_fbeta)
    # For Repeatedly cross-validating ML model
    cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
    return scores


def get_samplers():
    samplers = {
        # Under-samplers
        'RandomUn': RandomUnderSampler(),
        'TL': TomekLinks(),
        # 'ENN': EditedNearestNeighbours(),
        'RENN': RepeatedEditedNearestNeighbours(),
        'OSS': OneSidedSelection(),
        'NCR': NeighbourhoodCleaningRule(),
        'IHT': InstanceHardnessThreshold(),
        # Over-Samplers
        'RandomOv': RandomOverSampler(),
        'SMOTE': SMOTE(),
        'SMOTESVM': SVMSMOTE(),
        # 'SMOTEKMeans': KMeansSMOTE(),
        'ADASYN': ADASYN(),
        # Combined Under and Over Samplers
        'SMOTEENN': SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority')),
        'SMOTETomek': SMOTETomek(tomek=TomekLinks(sampling_strategy='majority')),
    }
    return samplers


def get_classifiers():
    classifiers = {
        "AdaBoost": AdaBoostClassifier(n_estimators=300),
        "Linear-SVM": SVC(kernel="linear", C=0.01),
        "RBF-SVM": SVC(gamma=1, C=10),
        "D-Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=11),
        "R-Forest": RandomForestClassifier(max_depth=10, max_features=4, n_estimators=300),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.5),
        "LDA": LinearDiscriminantAnalysis(),
        "MLP": MLPClassifier(alpha=0.001, max_iter=1000, hidden_layer_sizes=(50, 50)),
        "GaussNB": GaussianNB(),
        "LogReg": LogisticRegression(solver='liblinear'),
        "Ridge": RidgeClassifier(),
        "XGBoost": XGBClassifier(scale_pos_weight=99, n_estimators=500, max_depth=25),
        "Dummy(1)": DummyClassifier(strategy='constant', constant=1),
        "Dummy(0)": DummyClassifier(strategy='constant', constant=0),
    }
    return classifiers


def test_cost_sensitive_neural_net(X, y, num_splits=10, minority_cls_weight=10):
    from keras.layers import Dense
    from keras.models import Sequential

    def define_model():

        model = Sequential()
        # define first hidden layer and visible layer
        model.add(Dense(50, input_dim=np.shape(X)[1], activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(50, input_dim=50, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dense(100, input_dim=200, activation='relu', kernel_initializer='he_uniform'))
        # define output layer
        model.add(Dense(1, activation='sigmoid'))
        # define loss and optimizer
        model.compile(loss='binary_crossentropy', optimizer='sgd')
        return model

    start_time = time.time()
    scores = []
    skf = RepeatedStratifiedKFold(n_splits=num_splits, random_state=0, n_repeats=3)
    for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), desc="Folds", total=num_splits*3):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        model = define_model()
        history = model.fit(X_train, y_train, class_weight={0: 1, 1: minority_cls_weight}, epochs=100, verbose=0)
        y_pred = model.predict_classes(X_test)
        score = compute_fbeta(y_test, y_pred)
        scores.append(score)
    scores = np.array(scores)
    train_time = round(time.time()-start_time, 3)
    print("Model: {}, Avg Score: {} (+/- {}), Time Taken: {}s".format("MLP", round(np.mean(scores), 3),
                                                                      round(np.std(scores) * 2, 3), train_time))
