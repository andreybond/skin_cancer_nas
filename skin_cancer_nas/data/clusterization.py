#!/usr/bin/env python
# coding: utf-8

# In this notebook we are creating dataset for Melanoma classification.
# 
# 1. We are assigning spectre-tags to images (Red / Gree / InfraRed / White / UltraViolet).
# 2. We are adding clear tags - skin-bening / skin-melanoma / skin-clear / non-skin-ref (images of last type will be dropped fro from classification) 

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


import os
import pandas as pd
import ntpath
from pipeline_utils import *
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook
import time


# ## Load data 
# (flattened images information records)

# In[3]:


dir_path = os.path.abspath(os.path.join('..','..','data','interim','2_Melanoma_3_ch_1_ch_classification'))
df = pd.read_parquet(os.path.join(dir_path, 'df_images_stats_flat_fastparquet.gzip'), engine='fastparquet')
# df_lst.to_parquet(os.path.join(dir_path, 'df_lst_hists_freqs__pyarrow.parquet.gzip'), engine='pyarrow', compression='gzip')


# In[4]:


df.head()


# ## Clustering
# 
# We will drop img_type, as it seems badly separates images.
# We will use img_folder feature, as it is nicely separates lesions/raw_skin/references
# We have to decide about img_name features.

# We would like to run several clustering alternative runs and pick the 'best' one.

# In[5]:


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
            X_to_encode = X[self.cols_to_encode]
            df_non_encoded = X.drop(self.cols_to_encode, axis=1)
#             df_non_encoded = X.copy(deep=True).drop(self.cols_to_encode, axis=1)
            df_list_to_concat = [df_non_encoded] 
            
            for col_name_to_encode, prefix_to_use in zip(self.cols_to_encode, self.encoded_cols_prefixes):
                df_encoded = pd.get_dummies(X_to_encode[col_name_to_encode], prefix=prefix_to_use) 
                df_list_to_concat.append(df_encoded)
                
            return pd.concat(df_list_to_concat, axis=1)
            
        except Exception as e:
            print('DataFrameOneHotEncode error encountered: {}'.format(e))
            raise Exception(e)
        
    def fit(self, X, y=None, **fit_params):
        return self


# In[6]:


class DataFrameColDropper(BaseEstimator, TransformerMixin):
    """
    Class selects 'cols_to_select' from passed dataframe
    """
    
    def __init__(self, cols_to_drop):
        
        assert cols_to_drop is not None
        assert len(cols_to_drop) > 0
        
        self.cols_to_drop = cols_to_drop 
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        try:
            # we are doing 'C - D'
            C = list(X.columns)
            D = self.cols_to_drop
            self.cols_to_select = [item for item in C if item not in D]
            return X[self.cols_to_select]
            
        except KeyError:
            cols_error = list(set(self.cols_to_select) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
        
    def fit(self, X, y=None, **fit_params):
        return self


# In[7]:


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    Class applies passed in scaler to ALL columns of dataframe. Returns dataframe. 
    """
    
    def __init__(self, scaler):
        assert scaler is not None
        self.scaler = scaler       
        
    def transform(self, X):
#         assert isinstance(X, pd.DataFrame) 
        X[X.columns] = self.scaler.fit_transform(X[X.columns])
        return X
        
    def fit(self, X, y=None, **fit_params):
        return self


# In[8]:


class FitTransformWrapper(BaseEstimator, TransformerMixin):
    """
    Class selects 'cols_to_select' from passed dataframe
    """
    
    def __init__(self, tsne):
        assert tsne is not None
        self.tsne = tsne       
        
    def transform(self, X):
#         assert isinstance(X, pd.DataFrame)        
        return self.tsne.fit_transform(X)
        
    def fit(self, X, y=None, **fit_params):
        return self


# In[9]:


def tsnepipe__lapvar_hists_colors(dim_reduction):
    return Pipeline([
#         ('debug_1', Debug('1')),
        ('df_col_selector', DataFrameColDropper(['img_full_path', 'file_size_bytes', 'img_type', 'img_name_feature', 'img_folder_feature'])),
#         ('debug_2', Debug('2')),
        ("scale", MinMaxScaler(feature_range=(0, 1))),
#         ('debug_3', Debug('3')),
        ('dim_reduction', dim_reduction)
#         ('debug_4', Debug('4')),
    ])
def tsnepipe__lapvar_hists_colors_imgnamefolder(dim_reduction):
    return Pipeline([
        ('df_col_selector', DataFrameColDropper(['img_full_path', 'file_size_bytes', 'img_type'])),
        ('df_one_hot_encoder', DataFrameOneHotEncode(['img_name_feature', 'img_folder_feature'], ['img_name_feature_', 'img_folder_feature_'])),            
        ("scale", MinMaxScaler(feature_range=(0, 1))),
        ('dim_reduction', dim_reduction)
    ])
def tsnepipe__lapvar_imgnamefolder(dim_reduction):
    return Pipeline([
        ('debug_1', Debug('1')),
        ('df_col_selector', DataFrameColDropper(['img_full_path', 'file_size_bytes', 'img_type', 'dom_color_0', 'dom_color_1', 'dom_color_2', 'dom_color_3', 'dom_color_4', 'dom_color_freq_0', 'dom_color_freq_1', 'dom_color_freq_2', 'dom_color_freq_3', 'dom_color_freq_4'])),
        ('debug_2', Debug('2')),
        ('df_one_hot_encoder', DataFrameOneHotEncode(['img_name_feature', 'img_folder_feature'], ['img_name_feature_', 'img_folder_feature_'])),            
        ('debug_3', Debug('3')),
        ("scale", DataFrameScaler(MinMaxScaler(feature_range=(0, 1)))),
        ('debug_4', Debug('4')),
        ('dim_reduction', FitTransformWrapper(dim_reduction)),
        ('debug_5', Debug('5')),
    ])


# In[10]:


def tsnepipe__lapvar_hists(dim_reduction):
    return Pipeline([
        ('df_col_selector', DataFrameColDropper(['img_full_path', 'file_size_bytes', 'img_type', 'dom_color_0', 'dom_color_1', 'dom_color_2', 'dom_color_3', 'dom_color_4', 'dom_color_freq_0', 'dom_color_freq_1', 'dom_color_freq_2', 'dom_color_freq_3', 'dom_color_freq_4', 'img_name_feature', 'img_folder_feature'])),
        ("scale", DataFrameScaler(MinMaxScaler(feature_range=(0, 1)))),
        ('dim_reduction', dim_reduction)
    ])
def tsnepipe__lapvar_hists_imgnamefolder(dim_reduction):
    return Pipeline([
        ('df_col_selector', DataFrameColDropper(['img_full_path', 'file_size_bytes', 'img_type', 'dom_color_0', 'dom_color_1', 'dom_color_2', 'dom_color_3', 'dom_color_4', 'dom_color_freq_0', 'dom_color_freq_1', 'dom_color_freq_2', 'dom_color_freq_3', 'dom_color_freq_4'])),
        ('df_one_hot_encoder', DataFrameOneHotEncode(['img_name_feature', 'img_folder_feature'], ['img_name_feature_', 'img_folder_feature_'])),            
        ("scale", DataFrameScaler(MinMaxScaler(feature_range=(0, 1)))),
        ('dim_reduction', dim_reduction)
    ])


# In[11]:


def tsnepipe__lapvar_colors(dim_reduction):
    return Pipeline([
        ('df_col_selector', ColumnsSelector(['variance_of_laplace', 'dom_color_0', 'dom_color_1', 'dom_color_2', 'dom_color_3', 'dom_color_4', 'dom_color_freq_0', 'dom_color_freq_1', 'dom_color_freq_2', 'dom_color_freq_3', 'dom_color_freq_4'])),
        ("scale", DataFrameScaler(MinMaxScaler(feature_range=(0, 1)))),
        ('dim_reduction', dim_reduction)
    ])
def tsnepipe__lapvar_colors_imgnamefolder(dim_reduction):
    return Pipeline([
        ('df_col_selector', ColumnsSelector(['variance_of_laplace', 'dom_color_0', 'dom_color_1', 'dom_color_2', 'dom_color_3', 'dom_color_4', 'dom_color_freq_0', 'dom_color_freq_1', 'dom_color_freq_2', 'dom_color_freq_3', 'dom_color_freq_4', 'img_name_feature', 'img_folder_feature'])),
        ('df_one_hot_encoder', DataFrameOneHotEncode(['img_name_feature', 'img_folder_feature'], ['img_name_feature_', 'img_folder_feature_'])),  
        ("scale", DataFrameScaler(MinMaxScaler(feature_range=(0, 1)))),
        ('dim_reduction', dim_reduction)
    ])


# In[12]:


# tsne = TSNE(n_components=2, random_state=0, n_iter = 1000, min_grad_norm=0, verbose=1000)
# tsnepipe__lapvar_imgnamefolder(tsne).fit_transform(df.head(1000))


# In[13]:


# pipes = [tsnepipe__lapvar_hists_colors, tsnepipe__lapvar_hists_colors_imgnamefolder, tsnepipe__lapvar_imgnamefolder, tsnepipe__lapvar_hists, tsnepipe__lapvar_hists_imgnamefolder, tsnepipe__lapvar_colors, tsnepipe__lapvar_colors_imgnamefolder]

# prepared_pipelines = []
# for n_comp in range(2,3):
#     for perp in [100, 1000, 10000]:
#         for n in [250, 1000]:
#             for pipe in pipes:
#                 tsne_param_dict = {'n_components': n_comp, 'perplexity':perp, 'n_iter':n, 'verbose':1, 'n_jobs':34}
# #                 tsne = TSNE(*tsne_param_dict)
#                 tsne = TSNE(n_components=n_comp, perplexity=perp, n_iter=n, verbose=1, n_jobs=34)
#                 tsne_transform = MultiCoreTSNE(tsne)
#                 prepared_pipe = pipe(tsne_transform)
#                 prepared_pipelines.append((prepared_pipe, tsne_param_dict))


# ## Manual clusterization
# ### VarLapl + color dominance information

# In[14]:


df_var_domcol = ColumnsSelector(['variance_of_laplace', 'dom_color_0', 'dom_color_1', 'dom_color_2', 'dom_color_3', 'dom_color_4', 'dom_color_freq_0', 'dom_color_freq_1', 'dom_color_freq_2', 'dom_color_freq_3', 'dom_color_freq_4']).fit_transform(df)
df_var_domcol = DataFrameScaler(MinMaxScaler(feature_range=(0, 1))).fit_transform(df_var_domcol)


# In[15]:


df_var_domcol.head()


# In[16]:

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=1000, n_iter=100, n_jobs=34, random_state=0)
tsne_results = tsne.fit_transform(df_var_domcol.copy())
print('Multicore t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
print(tsne_results.shape())

# results_var_domcol = []

# for n_comp in range(2):
#     for perp in [200, 2000]:
#         for n in [250, 1000]:
#             tsne_param_dict = {'n_components': n_comp, 'perplexity':perp, 'n_iter':n, 'verbose':1, 'n_jobs':34, 'random_state':0}
# #                 tsne = TSNE(*tsne_param_dict)
#             time_start = time.time()
#             tsne = TSNE(n_components=n_comp, verbose=1, perplexity=perp, n_iter=n, n_jobs=34, random_state=0)
#             tsne_results = tsne.fit_transform(df_var_domcol.copy())
#             print('Multicore t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#             results.append((tsne_results, tsne_param_dict))


# In[ ]:





# In[ ]:



