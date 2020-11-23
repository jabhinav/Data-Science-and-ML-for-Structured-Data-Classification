from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from feature_engine import discretisers as dsc
import pandas as pd


class data_preprocess:

    def __init__(self, orig_data, label_col=None,
                 categorical_encoding='nominal', drop_columns=None):

        # Index of label columns should be computed after drop_columns are removed from the df
        # [orig_data.columns.get_loc(col) for col in drop_columns]
        self.drop_columns = drop_columns if drop_columns else []
        if label_col is None:
            label_col = orig_data.columns[-1]
        self.label_col = label_col
        self.orig_df = orig_data
        self.categorical_encoding = categorical_encoding
        self.y = None
        self.processed_df = None

        # Identify categorical/discrete/continuous features
        # make a list of categorical variables
        self.categorical = [
            var for var in orig_data.columns if orig_data[var].dtype == 'O' and (var not in drop_columns
                                                                                 and var != self.label_col)
        ]
        # make a list of numerical variables
        self.numerical = [
            var for var in orig_data.columns if orig_data[var].dtype != 'O' and (var not in drop_columns
                                                                                 and var != self.label_col)
        ]
        # From numerical make a list of discrete variables
        self.discrete = [var for var in self.numerical if len(orig_data[var].unique()) < 20]
        # continuous variables
        self.continuous = [var for var in self.numerical if var not in self.discrete]

    # def __call__(self, onehot_encode=None):
    #     return self.process_data(onehot_encode)


    @staticmethod
    def _drop_na(df, label_col):
        # Label column na filtering
        df = df[df[label_col].notna()]
        # Rows NA filtering
        df = df.dropna(axis=0,
                       how="all",
                       thresh=None,
                       subset=None,
                       inplace=False)
        # Columns NA filtering
        df = df.dropna(axis=1,
                       how="all",
                       thresh=None,
                       subset=None,
                       inplace=False)
        return df

    def get_df_X_y(self, df, label_column, onehot_encode=[], column_label_dict={}):
        if onehot_encode is None:
            onehot_encode = []
        df_Y = df[[label_column]]
        df_X = df.loc[:, df.columns != label_column]
        # char_cols = df_X.dtypes.pipe(lambda x: x[x == "object"]).index
        char_cols = self.categorical

        # Missing Data Imputation
        df_X[char_cols] = df_X[char_cols].fillna("")  # Will replace empty categorical feature values with "" (new cat)
        for col in self.numerical:
            df_X[col] = df_X[col].fillna(df_X[col].median())  # or use .mode()[0]
        # df_X.fillna(df_X.median(), inplace=True)# Since numerical columns are left with na, will fill them with median

        # One hot encoding of categorical variables that do not possess ordinal relationship.
        # If `columns` is None then all the columns with object or category dtype will be converted.
        # Nan or empty cells are provided separate 0-1 indicator.
        # onehot_encode = []
        if onehot_encode:
            onehot_encode = list(set(char_cols).intersection(onehot_encode))
            df_X = pd.get_dummies(df_X, prefix=[col[:4] for col in onehot_encode], columns=onehot_encode,
                                  prefix_sep='_', drop_first=False)

        # For each column, store its labels and use numbers to represent them (factorise)
        label_mapping = {}
        for c in char_cols:
            if c not in onehot_encode:
                # Factorise based on custom provided column label encoding
                # If label maps are provided for ordinal variables use that else use pre-defined method
                if c in column_label_dict.keys():
                    df_X[c].replace(column_label_dict[c], inplace=True)
                    label_mapping[c] = column_label_dict[c]
                else:
                    df_X[c], label_mapping[c] = pd.factorize(df_X[c], sort=True)

        label_cols = df_Y.dtypes.pipe(lambda x: x).index
        label_label_mapping = {}
        for c in label_cols:
            df_Y[c], label_label_mapping[c] = pd.factorize(df_Y[c])
        num_label_mapping = {
            num: label
            for num, label in enumerate(label_label_mapping[label_column])
        }
        # print("Category Label Mapping: {}".format(label_mapping))
        # print("Label Encoding: {}\n".format(num_label_mapping))

        # One hot encoding of categorical variables that do not possess ordinal relationship.
        # (handle_unknown='ignore'): If an unknown category is encountered during transform, the resulting one-hot
        # encoded columns for this feature will be all zeros .
        # if onehot_encode == "all_cat":
        #     # one hot encode cat features only
        #     ct = ColumnTransformer([('o', OneHotEncoder(handle_unknown='ignore'), self.categorical)],
        #                            remainder='passthrough')
        #     X = ct.fit_transform(df_X)
        # elif isinstance(onehot_encode, List):
        #     for feature in onehot_encode:
        #         if feature in self.categorical:
        #             pass
        #         else:
        #             print("Provided feature for one-hot not found in given dataset")
        #             sys.exit(0)
        #     ct = ColumnTransformer([('o', OneHotEncoder(), onehot_encode)], remainder='passthrough')
        #     X = ct.fit_transform(df_X)
        # else:
        #     X = df_X.values
        X = df_X.values
        y = df_Y.values.T[0]

        # Min-Max Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        self.y = df_Y
        self.processed_df = df_X
        return X, y

    def process_data(self):
        df = self.orig_df
        if self.drop_columns:
            df = df.drop(self.drop_columns, axis=1)
        # Drop empty columns/rows
        df = self._drop_na(df, self.label_col)
        return df


    @staticmethod
    def discretise_X(X, n_bins):
        bins = [n_bins for _ in range(X.shape[-1])]
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans').fit(X)
        X = est.transform(X)
        return X

    def discretise_using_decision_trees(self):
        # set up the discretisation transformer
        disc = dsc.DecisionTreeDiscretiser(
            cv=3,
            scoring='neg_mean_squared_error',  # Since, we are discretizing cont variables, use MSE to create splits
            variables=self.continuous,
            param_grid={'max_depth': [1, 2, 3, 4, 5]},
            regression=False)
        # fit the transformer
        disc.fit(self.processed_df, self.y)
        # transform and return the data
        discretized_X = disc.transform(self.processed_df)
        return discretized_X.values


"""
For categorical variables, treating nan(s) as another value of the variables is a reasonable approach.
For numerical variables, we will use missing-value imputation.

Note: Numerical Variables which have discrete values exhibit ordinal relationship if variable is treated as categorical.
The discrete numerical values of a feature are inherently ordered and thus, need no encoding. 
"""