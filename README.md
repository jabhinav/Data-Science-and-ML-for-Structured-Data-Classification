# Data-Science-and-ML-for-Structured-Data-Classification
Repo contains scripts to perform data analysis on structure data. It also provides comparison of various ML algorithms at different stages of data preparation.


## Overview
For the credit risk classification task, we will use EDA to explore data and infer some important relationships from it. We will then use these inferences to prepare data for model learning. We will follow up with an evaluation of multiple Machine Learning algorithms for binary classification and compare their performance. We will further address the imbalance / cost-sensitivity in our learning problem and finally, suggest some of the better performing models. It should be noted that this methodology is the perfect solution to the problem, but I hope to provide some better performing solutions and directions for future to further improve their performance.

## Data Preparation
Let’s take a look at the categorical variables in the dataset and see which exhibit ordinal relationship among its values. Following variables in the data are ordinal: Has been Employed for at least and at most, Savings Account Balance, Balance in existing bank account (lower and upper limit of the bucket). While, rest of the remaining categorical variables aren’t: Gender, Marital Status, Housing, Employment Status, Purpose, Property, Other EMI Plans and Loan History. For ordinal variables, we will employ Ordinal or Dummy encoding and One Hot for rest of them. Additionally, columns Telephone, Loan Application ID and Applicant ID are dropped based on the prior assumption that they do not contribute towards predicting the target variable.

## EDA
Refer attached jupyter notebook for an elaborate analysis of loan-applicant data.

## Key Observations
 Following observations can be made for the binary classification task:
- Imbalanced/Skewed Class Distribution: Only 300 out of 1000 samples are positive instances. Our positive class is under-represented.
- Requires Cost Sensitive Learning: This is because in the defined problem, missing a positive class (i.e. high-risk applicant) is worse than incorrectly classifying samples from negative class (i.e. low-risk applicants). In other words, cost incurred for false negatives is more than false positives. 
Imbalanced Classification and cost-sensitive learning are closely related. Specifically, we can address imbalanced learning using methods for cost-sensitive learning. The line between cost-sensitive augmentations to algorithms vs imbalanced classification augmentations to algorithms is blurred when the inverse of class distribution is used as the cost matrix.

## Baseline Model
A non-ML model which just always predicts 0 (i.e. a low-risk applicant) will obtain 70% accuracy. Based on the provided cost matrix, it will incur cost of 1*FPs + 4*FNs = 1*0 + 4*300 = 1200. Alternately, model always predicting 1 will obtain lower accuracy of 30% but a lower cost of 1*700 + 4*0 = 700. We need to do better on both fronts simultaneously. 

## Evaluation Metric
Use F2 score since False Negatives are costlier. Thus, we choose Recall over precision and this gives Beta greater than 1 in F-Beta score. We choose Beta of 2 to compute F-Beta. And since no separate dataset is provided for testing, k-fold (10-fold used for experiments) cross validation is used along with stratification to preserve class imbalance in each split.

## Machine Learning Algorithms for Binary Classification
I tried machine learning algorithms for binary classification in increasing order of complexity. For each selected ML model, I used Grid Search to determine best performing hyper-parameters in the specified range.
- Naïve Dummy: Predict minority class always. Will serve as a baseline, which any algorithm with skill has to surpass.
- Linear: Try Logistic Regression, LDA, Naïve-Bayes
- Non-Linear: Try Decision Tree, Neural Network based, SVM
- Ensemble: Random Forest (Bagging), Stochastic Gradient Boosting.

| Models | Stage1	| Stage2 | Stage3 | Stage4 |
| ------ | ------ | ------ | ------ | ------ |
| Dummy	| 0.682 (+/- 0.0)	| 0.682 (+/- 0.0)	| 0.682 (+/- 0.0)	| 0.682 (+/- 0.0) | 
| Logistic Regression	| 0.408 (+/- 0.17) | 0.645(+/- 0.116) |	0.689 (+/- 0.101) |	0.68 (+/- 0.111) |
| Ridge Regression	| 0.405(+/- 0.192)	| 0.646(+/- 0.122)| 	0.689 (+/- 0.096)| 	0.678 (+/- 0.108) |
| Naïve Bayes| 	0.55 (+/- 0.147) |	-	0.468 | (+/- 0.207)| 	- |
| LDA	0.417| (+/- 0.193)| 	-	| 0.69 (+/- 0.094)| 	- |
| QDA	0.067| (+/- 0.106)| 	-	| 0.657 (+/- 0.105)	| - |
| Linear-SVM| 	0.0 (+/- 0.0)| 	0.632(+/- 0.117)| 	0.701 (+/- 0.087)| 	0.666 (+/- 0.12) |
| RBF-SVM	| 0.137(+/- 0.114)| 	0.137(+/- 0.115)| 	0.697 (+/- 0.033)	| 0.697 (+/- 0.033) |
| MLP (50, 50)| 	0.421(+/- 0.188)	| 0.682(+/- 0.136)| 	0.679 (+/- 0.081)| 	- |
| Decision Tree	| 0.499(+/- 0.238)	| 0.655(+/- 0.132)| 	0.682 (+/- 0.101)| 	0.677 (+/- 0.096) |
| Random Forest	| 0.303(+/- 0.184)	| 0.51 (+/- 0.166)	| 0.709 (+/- 0.08)	| 0.702 (+/- 0.073) |
| Gradient Boosting	| 0.522(+/- 0.187)	| 0.626(+/- 0.166)	| 0.7 (+/- 0.083)	| 0.716 (+/- 0.072) |
| AdaBoost	| 0.485(+/- 0.131)	| -	| 0.696 (+/- 0.088)	| - |

Table 1: Evaluation of Machine Learning models in different stages. 

## Experiments/Ablation Study
Table 1, 2, 3 provide 10-fold F2 score of ML models. Experiments were conducted in stages to determine the better performing models.

### Data Preparation: 
Missing value data Imputation with median values, Min-Max normalization, one-hot categorical encoding for nominal variables. 

### Hyper-Parameter Selection: 
Best performing hyper-parameters for models were selected based on Grid Search with 5-fold cross validation. Best performing parameters of each split are provided in a json file enclosed inside ‘Results/hyperparam_selection’ folder. Best performing parameters of each split were further compared to determine overall best performing hyper-parameters of a model. Note that for below discussed results, we have conducted experiments after selecting best hyper-parameters for each model.

### Stage 1 (No under/over-sampling, no cost-sensitive learning): 
We can observe that none of the model is able to perform better than dummy classifier on the dataset with unbalanced distribution. Nonetheless, Naïve Bayes with independent feature assumption is performing the best. Let’s try cost-sensitive learning for some algorithms and see if there is any improvement.

### Stage 2 (No under/over-sampling, cost-sensitive learning): 
Configure algorithms so that misclassification cost is inversely proportional to distribution of examples in the training dataset. Used “balanced” weights for classes in Logistic/Ridge Regression, SVM, Decision Tree, Random Forest and “scale_pos_weight=0.99” for Gradient Boosting. With cost-sensitivity, we can observe huge boost across algorithms, most-notably for linear classifiers like SVM, and Logistic Regression. For MLP, we used keras model and class weights to promote cost-sensitive learning. Score increased to 0.66 (+/- 0.155) (1:4 weight ratio), reduced to 0.603 (+/- 0.147) with 1:2 weight ratio and further increased to 0.682 (+/- 0.136) with 1:13 weight ratio. No further improvement is observed when weight to minority class was increased. However, models have not still acquired skill to surpass our dummy classifier. Let’s try if sampling affects the performance of classifier.

### Stage 3 (under/over-sampling, no cost-sensitive learning):
To address imbalanced learning, whether to use under/over sampling data with Logistic Regression (linear model) and Random Forest model (non-linear and ensemble model) refer Table 2 and 3. It can be observed that over-sampling from minority class can lead to over-fitting on minority class which leads to poor performance on hold-out dataset. For example, performance of logistic regression with over-sampling has not improved. Additionally, for Random Forest the performance of the model with over-sampling has although increased but it is still not enough to surpass the dummy model. Both algorithms tend to benefit more from certain under-sampling techniques like Repeated ENN and Instance Hardness Threshold, able to surpass the dummy classifier. We can use under-sampling since the training distribution is not heavily skewed and offers sufficient number of examples in minority class. Since, we are prioritizing predictions on minority class, deleting samples from majority class for balanced distribution seems to have little to no loss of important information required for making predictions. Repeated Edited Nearest Neighbors method was chosen amongst other over and under sampling methods for its high performance and less computation time. With under-sampling method Repeated Edited Nearest Neighbors for majority class, evaluated various ML models in Table 1.

### Stage 4 (under sampling and cost-sensitive learning): 
For XGBoost, performance is 0.702 (+/- 0.077) with weight=2, 0.707 (+/- 0.071) with weight=4, 0.713 (+/- 0.07) with weight=10, 0.716 (+/- 0.069) with weight = 99 (suggested heuristic). With both types of learning, performance improvement is observed only for XGBoost. For remaining, performance has either remained same or decreased slightly. 


|Model	|F2 Score	|Computation Time|
|-------|---------|----------------|
|Dummy	|0.682 (+/- 0.0)|	-|
|LogReg	|0.645 (+/- 0.116)	|0.12s|
|Under-Sampling|
|LogReg + Random | 0.633 (+/- 0.135)|	0.09s|
|LogReg + Tomek Links	| 0.641 (+/- 0.125)|	0.56s|
|LogReg + Repeated ENN	| 0.68 (+/- 0.111)|	0.72s|
|LogReg + One-Sided Selection	| 0.64 (+/- 0.122)|	1.11s|
|LogReg + Nearest Neighborhood Cleaning Rule	| 0.645 (+/- 0.119)|	0.97s|
|LogReg + Instance Hardness Threshold	| 0.698 (+/- 0.088)|	20.14s|
|Over-Sampling|
|LogReg + Random	|0.646 (+/- 0.12)|	0.13s|
|LogReg + SMOTE	|0.636 (+/- 0.118)|	0.21s|
|LogReg + SMOTESVM	|0.62 (+/- 0.147)|	1.19s|
|LogReg + ADSYN	|0.644 (+/- 0.136)|	0.41s|
|Combined|
|LogReg + SMOTETomek	|0.623 (+/- 0.121)|	1.05s|
|LogReg + SMOTEENN	|0.666 (+/- 0.11)|	0.69s|

Table 2: Sampling performance on Logistic Regression Model

|Model	|F2 Score	|Computation Time|
|-------|---------|----------------|
|Dummy	|0.682 (+/- 0.0)	|-|
|RForest|	0.496 (+/- 0.181)	|12.17s|
|Under-Sampling|
|RForest + Random	|0.676 (+/- 0.099)	|11.28s|
|RForest + Tomek Links	|0.549 (+/- 0.17)	|13.34s|
|RForest + Repeated ENN	|0.704 (+/- 0.086)	|11.15s|
|RForest + One-Sided Selection	|0.556 (+/- 0.191)	|13.03s|
|RForest + Nearest Neighborhood Cleaning Rule|	0.684 (+/- 0.113)	|11.81s|
|RForest + Instance Hardness Threshold	|0.71 (+/- 0.07)	|31.09s|
|Over-Sampling|
|RForest + Random	|0.579 (+/- 0.153)	|13.76s|
|RForest + SMOTE	|0.539 (+/- 0.147)|	15.22s|
|RForest + SMOTESVM	|0.533 (+/- 0.168)	|16.23s|
|RForest + ADSYN	|0.545 (+/- 0.153)	|15.57s|
|Combined|
|RForest + SMOTETomek	|0.543 (+/- 0.163)	|16.35s|
|RForest + SMOTEENN	|0.707 (+/- 0.081)|	13.42s|

Table 3: Sampling performance on Random Forest Model


## (Future Work) Label Noise: 
We can also try correcting noisy labels to improve model performance since, mislabelled minority class instances will contribute to increasing the perceived imbalance ratio, as well as introduce mislabelled noisy instances inside the class region of the minority class. On the other hand, mislabelled majority class instances may lead the learning algorithm, or imbalanced treatment methods, focus on wrong areas of input space.

