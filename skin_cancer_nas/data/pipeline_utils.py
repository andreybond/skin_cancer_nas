import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class MatrixToPandas(TransformerMixin, BaseEstimator):
    """
    Class accepts 2D array and splits it into Pandas dataframe
    with single column holding rows with lists of 1D arrays

    NB.: Subject to change in the future, use carefully.
    """

    def __init__(self, colname):
        self.colname = colname

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2

        rows_count = X.shape[0]
        list = [x.flatten() for x in np.split(X, rows_count)]
        return pd.DataFrame({self.colname: list})


class PandasTransform(TransformerMixin, BaseEstimator):
    """
    This class will not change.
    """

    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return self.fn(X)


class PandasFeatureUnion(FeatureUnion):
    """
    This class will not change.
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


class DummyTransformer(TransformerMixin):
    """
    Just a dummy transformer to understand Pipeline API behaviour.

    NB.: Subject to change/remove in the future, use carefully.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
    Class accepts pd.DataFrame and allows to select columns

    Returns:  pd.DataFrame
    """

    # Seems working (taken from: https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/)
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class SeriesSelector(BaseEstimator, TransformerMixin):
    """
    Class accepts pd.DataFrame and allows to select single column as pd.Series

    NB.: Subject to change/remove in the future, use carefully.
    """

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            _series = X[self.column]
            return _series
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class DataPrinter(TransformerMixin):
    """
    Pipelines debugging class, dows not transform data, just prints type, head, dimensions.

    NB.: Can be dropped in favor of Debug class
    """

    def __init__(self, prefix):
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('------------------------------')
        print('--' + str(self.prefix) + '--')
        print(X.shape)
        return X


class Debug(BaseEstimator, TransformerMixin):
    """
    Pipelines debugging class, does not transform data, just prints type, head, dimensions.
    """

    def __init__(self, prefix='0'):
        self.prefix = prefix

    def transform(self, X):
        print('------------------------------')
        print('--' + str(self.prefix) + '--')
        print(type(X))
        if isinstance(X, np.ndarray):
            print(pd.DataFrame(X).head())
            print(X.shape)
        elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            print(X.head())
            print(X.shape)
        elif isinstance(X, scipy.sparse.csr.csr_matrix):
            print(pd.DataFrame(X.toarray()).head())
            print(X.shape)
        elif isinstance(X, list):
            print(pd.DataFrame({'column': X}).head())
            print(len(X))
        # elif isinstance(X, generator):
        #     print('X is generator.')
        else:
            print('Unsupported X type = ' + str(
                type(X)) + ', should be np.ndarray, scipy.sparse.csr.csr_matrix, generator, pd.Series or pd.DataFrame!')
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class DataFrameAsMatrix(BaseEstimator, TransformerMixin):
    """
    pd.DataFrame transformator to nd.array

    NB: can be dropped because PandasTransform(pd.as_matrix) seems provides this functionality (To Be Tested)
    """

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.as_matrix()

    def fit(self, X, y=None, **fit_params):
        return self
    
class DataFrameToNumpy(BaseEstimator, TransformerMixin):
    """
    pd.DataFrame transformator to nd.array

    NB: can be dropped because PandasTransform(pd.as_matrix) seems provides this functionality (To Be Tested)
    """

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.as_matrix()

    def fit(self, X, y=None, **fit_params):
        return self

    
class DataFrameOneHotEncode(BaseEstimator, TransformerMixin):
    """
    Class accepts 'columns_to_encode', 'encoded_cols_prefixes' and one_hot_encodes these columns for passed dataframe.  
    """
    
    def __init__(self, cols_to_encode, encoded_cols_prefixes):
        
        assert cols_to_encode is not None
        assert encoded_cols_prefixes is not None
        assert isinstance(cols_to_encode, list)
        assert isinstance(encoded_cols_prefixes, list)
        assert len(cols_to_encode) == len(encoded_cols_prefixes)
        
        self.cols_to_encode = cols_to_encode 
        self.encoded_cols_prefixes = encoded_cols_prefixes
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        try:
            df_non_encoded = X.copy(deep=True).drop(self.cols_to_encode, axis=1)
            df_list_to_concat = [df_non_encoded] 
            
            for col_name_to_encode, prefix_to_use in zip(self.cols_to_encode, self.encoded_cols_prefixes):
                df_encoded = pd.get_dummies(X[col_name_to_encode], prefix=prefix_to_use) 
                df_list_to_concat.append(df_encoded)
                
            return pd.concat(df_list_to_concat, axis=1)
            
        except Exception as e:
            print('DataFrameOneHotEncode error encountered: {}'.format(e))
            raise Exception(e)
        
    def fit(self, X, y=None, **fit_params):
        return self
    
     