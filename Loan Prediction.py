#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


os.chdir("C:\\Users\\91721\\Desktop\\Loan prediction")


# In[3]:


train=pd.read_csv("train_ctrUa4K.csv")


# In[4]:


train.head()


# In[5]:


train.describe()


# In[9]:


train.isnull().sum()


# In[13]:


train.dtypes


# In[16]:


train.shape


# In[18]:


train.isnull().sum()


# In[19]:


train['Gender'].value_counts()


# In[20]:


train.Gender=train.Gender.fillna('Male')


# In[23]:


train['Married'].value_counts()


# In[24]:


train.Married=train.Married.fillna('Yes')


# In[25]:


train['Dependents'].value_counts()


# In[26]:


train.Dependents=train.Dependents.fillna('0')


# In[27]:


train['Self_Employed'].value_counts()


# In[28]:


train.Self_Employed=train.Self_Employed.fillna('No')


# In[29]:


train.LoanAmount=train.LoanAmount.fillna(train.LoanAmount.mean())


# In[30]:


train['Loan_Amount_Term'].value_counts()


# In[31]:


train.Loan_Amount_Term=train.Loan_Amount_Term.fillna('360.0')


# In[32]:


train['Credit_History'].value_counts()


# In[33]:


train.Credit_History=train.Credit_History.fillna('1')


# In[35]:


train.isnull().sum()


# In[37]:


train.boxplot('ApplicantIncome')


# In[38]:


train.boxplot(column='ApplicantIncome', by = 'Education')


# In[39]:


train['LoanAmount'].hist(bins=20)


# In[40]:


# Perform log transformation of TotalIncome to make it closer to normal
train['LoanAmount_log'] = np.log(train['LoanAmount'])

# Looking at the distribtion of TotalIncome_log
train['LoanAmount_log'].hist(bins=20)


# In[ ]:




