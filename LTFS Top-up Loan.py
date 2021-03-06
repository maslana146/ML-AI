#!/usr/bin/env python
# coding: utf-8

# ## Analytics Vidya LTFS FinHack 3
# About this data
# 
# LTFS provides it’s loan services to its customers and is interested in selling more of its Top-up loan services to its existing customers.
# 
# Develop a model for the interesting business challenge ‘Upsell Predictions'
# Content
# 
#     Customer’s Demographics: The demography table along with the target variable & demographic information contains variables related to Frequency of the loan, Tenure of the loan, Disbursal Amount for a loan & LTV.
# 
#     Bureau data: Bureau data contains the behavioural and transactional attributes of the customers like current balance, Loan Amount, Overdue etc. for various tradelines of a given customer
#  Business Objective: Predict when to pitch a Top-up during the original loan tenure.\
#  Problem Statement: It's multilabel classification.
# 
# 

# In[1]:


import pandas as pd
import pandas_profiling as pdp
import os
import numpy as np
from zipfile import ZipFile
import re,gc
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import prince
from category_encoders import wrapper,target_encoder
from scipy.stats import boxcox_normmax,norm,kurtosis, skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from optbinning import BinningProcess
from sklearn.preprocessing import OneHotEncoder,StandardScaler,Normalizer
from sklearn.decomposition import PCA


# In[2]:


def load_data(zip_fn):
    zf = ZipFile(zip_fn)
    filenames = [i for i in zf.namelist() if re.match('.*(?<!submission\.)csv$',i)]
    return [pd.read_csv(zf.open(filename)) for filename in filenames]


# In[3]:


demographics_dictionary,test,test_bureau,train,train_bureau =load_data('data.zip')


# In[4]:


train = train.drop_duplicates()
test  = test.drop_duplicates()
train_bureau = train_bureau.drop_duplicates()
test_bureau  = test_bureau.drop_duplicates()
df = pd.concat([train,test])
bureau = pd.concat([train_bureau, test_bureau])
del train_bureau,test_bureau,train,test
gc.collect()


# Let's do some magic with cyclical features. The explanation with some deeper insight can be found in <a href="http://blog.davidkaleko.com/feature-engineering-cyclical-features.html">here</a>
# .

# In[5]:


date_attr = ['AuthDate','DisbursalDate','MaturityDAte']
for i in date_attr:
    df.loc[:,i] = pd.to_datetime(df[i])
    df[f'{i[:-4]}'+'_day_sin'] =  np.sin((df[i].dt.day-1)*(2.*np.pi/31))
    df[f'{i[:-4]}'+'_day_cos'] = np.cos((df[i].dt.day-1)*(2.*np.pi/31))
    df[f'{i[:-4]}'+'_month_sin'] = np.sin((df[i].dt.month-1)*(2.*np.pi/12))
    df[f'{i[:-4]}'+'_month_cos'] = np.cos((df[i].dt.month-1)*(2.*np.pi/12))
    df[f'{i[:-4]}'+'_year'] = df[i].dt.year
df['loan_approval_days'] = (df['AuthDate'] - df['DisbursalDate']).dt.days
df['loan_period'] = (df['MaturityDAte'] - df['DisbursalDate']).dt.days
df.drop(['AuthDate','DisbursalDate','MaturityDAte','AssetID'],axis=1,inplace=True)


# Split columns into categories

# In[6]:


cat_attr = ['Frequency','InstlmentMode','LoanStatus','PaymentMode','ManufacturerID','SEX','State','Maturity_year','Disbursal_year','Auth_year']
high_cardinality_cat_attr = ['BranchID','Area','SupplierID','City','ZiPCODE']
cat_attr = list(set(cat_attr) - set(high_cardinality_cat_attr))
num_attr = list(set(df.columns) - set(cat_attr) - set(date_attr)- set(high_cardinality_cat_attr) - set(['AuthDate','DisbursalDate','MaturityDAte','AssetID','ID','Top-up Month']))


# df.columns == attributes + ID + Top-up Month

# In[9]:


assert len(cat_attr+high_cardinality_cat_attr+num_attr)+2 == df.columns.size


# In[10]:


def draw_plots_categorical(grouped_by,target,df,percentage_threshold):
    temp = pd.DataFrame()
    unique = pd.unique(df[grouped_by].dropna())
    for attr in unique:
        temp[attr] = df[df[grouped_by]==attr][target].value_counts()
    columns = dict(zip(list(range(len(temp.columns))),temp.columns))
    temp = temp.rename(columns=columns)
    return get_plots_categorical(temp,grouped_by,target,percentage_threshold)


# In[11]:


def get_plots_categorical(temp,attribute,target,percentage_threshold):
    fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.3],
                    specs=[[ {"colspan": 2},{}],
                        [{"type": "xy"}, {"type": "pie"}]])
    pie_temp = temp.sum()
    probs = pie_temp/sum(pie_temp)*100
    thresh_temp = probs[probs>=percentage_threshold]
    other = probs[probs<percentage_threshold]
    if other.size >1:
        thresh_temp['other'] = sum(other)
        temp['other'] = temp.loc[:,other.index].sum(axis=1)
        temp.drop(other.index,axis = 1,inplace=True)
    else:
        thresh_temp = probs
    for counter,i in enumerate(temp.index):
        fig.add_trace(go.Bar(y=temp[temp.index==i].values[0],x=temp.columns,name=i,
                         legendgroup='group1',marker_color=px.colors.qualitative.Prism[counter]), row=1, col=1)
    temp = temp.T
    for counter,i in enumerate(temp.index):
        fig.add_trace(go.Bar(y=temp[temp.index==i].values[0],x=temp.columns,name=i,
                    legendgroup='group1',showlegend=False,marker_color=px.colors.qualitative.Prism[counter%2]), row=2, col=1)
    fig.add_trace(go.Pie(labels=thresh_temp.index,values=thresh_temp,legendgroup='group2',marker=dict(colors= px.colors.qualitative.Prism)),row =2,col=2)
    fig.layout.update(barmode='group',
        height=800, legend=dict(tracegroupgap = 250),title=f'{attribute} vizualization based on {target}')
    fig.show()


# The last attribute is the percentage threshold. I decided to plot categories which percentage of the occurance is higher than the threshold. The rest is combined to "other" category.

# In[13]:


class LabelCombiner(BaseEstimator, TransformerMixin):
    def __init__(self,percentage_threshold=3):
        self.percentage_threshold = percentage_threshold
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for i in range(X.shape[1]):
            arr =X[:,i]
            if arr.dtype == np.float64:
                arr.dtype = np.int32
            labels,count = np.unique(arr, return_counts=True)
            count = (count/count.sum())*100
            categories = labels[count > self.percentage_threshold]
            X[:,i] = np.where(~np.isin(arr,categories), 'other', arr) 
        return X


# In[14]:


class SkewnessFixer(BaseEstimator, TransformerMixin):
    def __init__(self,threshold = 0.5,exclude= [None]):
        self.threshold = threshold
        self.exclude = exclude
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric = []
        skew_features = []
        for i in range(X.shape[1]):
            if X[:,i].dtype in numeric_dtypes:
                if i not in self.exclude:
                    skewness = abs(skew(X[:,i]))
                    if skewness > self.threshold:
                        skew_features.append(i)
        for i in skew_features:
            if min(X[:,i]) > 0:
                X[:,i] = boxcox1p(X[:,i], boxcox_normmax(X[:,i] + 1))
        return X


# Alternative for dummy encoding. Handling high cardinality data by encoding attributes in n (number of classes in multiclass predicition) attributes by target encoding. Do not know whether it will work. In case it does not we have two alternatives:
# * drop attributes with high cardinality to prevent negative impact on future models
# * encode attributes by dummy encoding and hoping that it will somehow work

# In[15]:


encoder = wrapper.PolynomialWrapper(target_encoder.TargetEncoder())


# In[16]:


num_pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('skewness_fixer', SkewnessFixer()),
            ('normalizer', Normalizer()),
            ('std_scaler', StandardScaler())
            ])


# In[17]:


cat_pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('label combiner',LabelCombiner()),
            ('OHE', OneHotEncoder())
])


# In[18]:


high_card_cat = Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),   
            ("high_cardinality_cat",encoder)
])


# In[19]:


preprocessing_pipeline =  ColumnTransformer([
            ("num", num_pipeline, num_attr),
            ("cat", cat_pipeline, cat_attr),
            ("high_cardinality_cat",high_card_cat, high_cardinality_cat_attr)
            ])


# In[62]:


train = df[~df['Top-up Month'].isna()]
train.loc[:,cat_attr] = train.loc[:,cat_attr].astype(str)
X = train.drop(columns = ["ID"])
y = train[['Top-up Month']]


# In[22]:


train = preprocessing_pipeline.fit_transform(X,y)


# In[41]:


pre_train = pd.DataFrame(train)


# In[42]:


bool_cols = [col for col in pre_train 
             if np.isin(pre_train[col].dropna().unique(), [0, 1]).all()]
num_cols = list(set(pre_train.columns)-set(bool_cols))


# We can see that having 15 components in enough to describe all numerical columns

# In[44]:


pca = PCA().fit(pre_train[num_cols])
x = np.cumsum(pca.explained_variance_ratio_)
fig = px.line( y=x, x=list(range(len(x))))
fig.update_layout(title='PCA algorithm',
                   xaxis_title='number of components',
                   yaxis_title='cumulative explained variance')
fig.show()


# In[45]:


PCA_cols = PCA(n_components=15).fit_transform(pre_train[num_cols])
pre_train.drop(columns = num_cols,inplace=True)
pre_train = pd.concat([pre_train,pd.DataFrame(PCA_cols)],axis=1)


# In[47]:


pre_train


# In[ ]:




