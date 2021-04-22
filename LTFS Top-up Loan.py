#!/usr/bin/env python
# coding: utf-8

# ## Vizaulization TODO:
# * come up with idea on who to visualize categorical attributes. Should discretization be done beforehand?
# * find out who can one obtain useful knowledge from date attributes, how to visualize it?
# * analyze and iterpret numerical attributes
# 

# In[1]:


import pandas as pd
import pandas_profiling as pdp
import os
import numpy as np
from zipfile import ZipFile
import re
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# In[2]:


def load_data(zip_fn):
    zf = ZipFile(zip_fn)
    filenames = [i for i in zf.namelist() if re.match('.*(?<!submission\.)csv$',i)]
    return [pd.read_csv(zf.open(filename)) for filename in filenames]


# In[3]:


demographics_dictionary,test,test_bureau,train,train_bureau =load_data('data.zip')


# In[4]:


train.info()


# In[5]:


cols = train.columns
num_cols = train._get_numeric_data().columns
categorical_cols = list(set(cols) - set(num_cols))
categorical_cols = [i for i in categorical_cols if re.match('^(?!(TOP-UP MONTH|.*DATE)$).*$',i.upper())]
print(categorical_cols)


# In[6]:


def draw_plots_categorical(grouped_by,target,df):
    temp = pd.DataFrame()
    unique = pd.unique(df[grouped_by].dropna())
    for attr in unique:
        temp[attr] = df[df[grouped_by]==attr][target].value_counts()
    columns = dict(zip(list(range(len(temp.columns))),temp.columns))
    temp = temp.rename(columns=columns)
    return get_plots_categorical(temp,grouped_by,target)


# In[7]:


def get_plots_categorical(temp,attribute,target):
    fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.3],
                    specs=[[ {"colspan": 2},{}],
                        [{"type": "xy"}, {"type": "pie"}]])
    for counter,i in enumerate(temp.index):
        fig.add_trace(go.Bar(y=temp[temp.index==i].values[0],x=temp.columns,name=i,
                         legendgroup='group1',marker_color=px.colors.qualitative.Prism[counter]), row=1, col=1)
    pie_temp = temp.sum()
    temp = temp.T
    for counter,i in enumerate(temp.index):
        fig.add_trace(go.Bar(y=temp[temp.index==i].values[0],x=temp.columns,name=i,
                    legendgroup='group1',showlegend=False,marker_color=px.colors.qualitative.Prism[counter%2]), row=2, col=1)
    fig.add_trace(go.Pie(labels=pie_temp.index,values=pie_temp,legendgroup='group2',marker=dict(colors= px.colors.qualitative.Prism)),row =2,col=2)
    fig.layout.update(barmode='group',
        height=800, legend=dict(tracegroupgap = 250),title=f'{attribute} vizualization based on {target}')
    fig.show()


# In[8]:


draw_plots_categorical('LoanStatus','Top-up Month',train)


# In[9]:


draw_plots_categorical('Frequency','Top-up Month',train)


# In[10]:


draw_plots_categorical('PaymentMode','Top-up Month',train)


# In[11]:


corr = train.corr().abs().unstack().sort_values(kind='quicksort',ascending=False)
corr = pd.DataFrame(data=corr[corr != 1]).reset_index(inplace=False) 
corr["pair"] = corr["level_0"] + " - " + corr["level_1"]
corr = corr.drop(columns=["level_0","level_1"]).drop_duplicates(subset=[0]).rename(columns = {0:'value'}, inplace = False)
corr


# In[12]:



px.bar(corr.loc[corr['value'] > 0.3], x='value',y="pair",color='value')


# In[16]:


lasso_train = train.drop(columns=["LTV"])
for col in lasso_train[lasso_train.loc[:, lasso_train.dtypes == object].columns]:
    lasso_train[col] = lasso_train[col].astype('category').cat.codes
lasso_train = lasso_train.fillna(0)
target = train["LTV"]
lasso_train


# In[17]:


reg = LassoCV()
reg.fit(lasso_train, target)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(lasso_train,target))
coef = pd.Series(reg.coef_, index = lasso_train.columns)


# In[18]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[32]:


imp_coef = pd.DataFrame(coef.sort_values()).rename(columns={0:'value'})
px.bar(imp_coef,orientation='h',title="FInfluence of attributes on LTV")


# >`LTV` - Loan to value (borrowed amount against the appraised value of the property )
