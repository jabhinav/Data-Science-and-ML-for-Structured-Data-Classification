import os
import time
import pandas as pd
import numpy as np
import sys
import warnings
from data_reader import LoadCSV
from data_preprocess import data_preprocess
from utils import box_plot
from models import evaluate_model, get_classifiers, get_samplers, run_classifiers_kfold_with_grid_search, \
    test_cost_sensitive_neural_net
from matplotlib import pyplot
from imblearn.pipeline import Pipeline


pyplot.rcParams.update({'font.size': 6})

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def test_samplers_with_clfs(clfs, samplers, X, y, num_splits=5, make_plots=False):
    """
    :param clfs: Dictionary of classifiers. Refer get_classifiers()
    :param samplers: Dictionary of samplers. Refer get_samplers()
    :param X: Input Data
    :param y: Target
    :param num_splits: Number of Splits for cross validation
    :param make_plots: (bool) to make box-plot which provides visualisation of model's performance on each split
    """
    results = []
    model_names = []
    for clf in clfs.keys():
        for sampler in samplers.keys():
            model_names.append(clf + "_" + sampler)
            start = time.time()
            model = Pipeline(steps=[
                ('s', samplers[sampler]),
                ('m', clfs[clf])
            ])
            scores = evaluate_model(X, y, model, num_splits=num_splits)
            train_time = round(time.time() - start, 2)
            print("Model: {}, Avg Score: {} (+/- {}), Time Taken: {}s".format(clf + "_" + sampler,
                                                                              round(scores.mean(), 3),
                                                                              round(scores.std() * 2, 3),
                                                                              train_time))
            results.append(scores)
    if make_plots:
        box_plot(results, model_names, "x", save_fig=False)


# noinspection PyShadowingNames
def main(path_to_applicant_data, path_to_loan_data):

    # Read Data
    start = time.time()
    applicant_data_df = LoadCSV(data_path=path_to_applicant_data)()
    loan_data_df = LoadCSV(data_path=path_to_loan_data)()
    print("Time taken to load data: {}".format(round(time.time() - start, 3)))

    # Merge applicant data with loan data on applicant_id
    start = time.time()
    data_df = pd.merge(applicant_data_df, loan_data_df, on='applicant_id')
    print("Time taken to merge data: {}".format(round(time.time() - start, 3)))

    # Feature Selection Based on EDA
    drop_secondary = False
    pre_select_features = False
    label_col = {'high_risk_applicant'}
    primary_features = ['Months_loan_taken_for', 'Principal_loan_amount', 'Other_EMI_plans',
                        'Number_of_existing_loans_at_this_bank', 'Loan_history']
    secondary_features = ['Employment_status', 'Has_been_employed_for_at_least', 'Has_been_employed_for_at_most',
                          'Foreign_worker', 'Primary_applicant_age_in_years', 'Has_guarantor']
    no_info_features = ['applicant_id', 'loan_application_id', 'Telephone']
    if drop_secondary and pre_select_features:
        drop_features = list(set(data_df.columns) - set(primary_features) - label_col)
    elif not drop_secondary and pre_select_features:
        drop_features = list(set(data_df.columns) - set(primary_features) - set(secondary_features) - label_col)
    else:
        drop_features = no_info_features

    # # Pre-process data
    start = time.time()
    column_label_dict = {
        'Savings_account_balance': {
            '': 0,
            'Low': 1,
            'Medium': 2,
            'High': 3,
            'Very high': 4
        }
    }
    feature_req_onehot_enc = ['Gender', 'Marital_status', 'Housing', 'Purpose', 'Property',
                              'Employment_status', 'Loan_history', 'Other_EMI_plans']
    data_preprocess_obj = data_preprocess(data_df, drop_columns=drop_features)
    processed_df = data_preprocess_obj.process_data()
    # Note implemented one-hot considers missing values of categorical variables as separate class.
    # For the given data, onehot_encode= List[<ColName>], None
    X, y, = data_preprocess_obj.get_df_X_y(processed_df, label_column='high_risk_applicant',
                                           onehot_encode=feature_req_onehot_enc,
                                           column_label_dict=column_label_dict)
    print("Time taken to process data: {}".format(round(time.time() - start, 3)))

    n_samples, n_features = np.shape(X)
    print("Number of instances: {}\nNumber of features: {}".format(n_samples, n_features))
    # Check for class imbalance
    print("Number of  Positive/Negative class instances: {}/{}".format(sum(y), len(y)-sum(y)))

    # To identify best-performing hyper-parameters
    # run_classifiers_kfold_with_grid_search(X, y, result_dir="./Results/hyperparam_selection/", num_splits=5, keyword='')

    # # Define Model with best-hyperparam first and then evaluate
    print("\nEvaluating Models...")
    test_samplers_with_clfs(get_classifiers(), {"RENN": get_samplers()['RENN']}, X, y, num_splits=10, make_plots=True)

    # # For selective model evaluation, specify the model name (refer models.py) or multiple names for evalaution.
    # model_names = ['R-Forest']
    # test_samplers_with_clfs({model: get_classifiers()[model] for model in model_names},
    #                         {"RENN": get_samplers()['RENN']}, X, y, num_splits=10, make_plots=False)
    # test_cost_sensitive_neural_net(X, y, num_splits=10, minority_cls_weight=13)


if __name__ == '__main__':
    path_to_applicant_data = "./applicant.csv"
    path_to_loan_data = "./loan.csv"
    main(path_to_applicant_data, path_to_loan_data)
