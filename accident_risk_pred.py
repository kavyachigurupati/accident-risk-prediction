#!/usr/bin/env python
# coding: utf-8

# In[12]:


import polars as pl
from polars import selectors as cs
import lightgbm as lgb


# In[26]:


from lightgbm import early_stopping
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import roc_auc_score


# In[5]:


data = pl.read_csv("playground-series-s5e10/train.csv")


# In[7]:


train = pl.DataFrame(data)


# In[11]:


string_cols = train.select(cs.string()).columns
print(string_cols)


# In[14]:


read_test_data = pl.read_csv("playground-series-s5e10/test.csv")
test = pl.DataFrame(read_test_data)
print(test)


# In[15]:


train = train.with_columns(
    cs.string().cast(pl.Categorical).to_physical(),
    cs.boolean().cast(pl.Int8)
)

test = test.with_columns(
    cs.string().cast(pl.Categorical).to_physical(),
    cs.boolean().cast(pl.Int8)
)


# In[17]:


X = train.drop('accident_risk')
y = train.select("accident_risk")
print(X.shape)
print(y.shape)


# In[21]:


from sklearn.model_selection import train_test_split

# Split data (e.g. 80% train, 20% validation)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_cols_indices = [X.columns.index(col) for col in string_cols]
# Convert to LightGBM Dataset format
train_data = lgb.Dataset(X_train, label=y_train.to_numpy().flatten(), categorical_feature=categorical_cols_indices)
valid_data = lgb.Dataset(X_valid, label=y_valid.to_numpy().flatten(), reference=train_data)

# Define parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}


# In[70]:


# train_data = lgb.Dataset(X_train, y_train.to_numpy().flatten(), )
# Train the model
bst = lgb.train(
    params,
    train_data,
    num_boost_round=100
    # valid_sets=[valid_data],
    # callbacks=[early_stopping(10)]  # 👈 optional but recommended
)


# In[ ]:


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}



# In[33]:


with mlflow.start_run() as run:
    
    # Train model
    bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data], callbacks=[early_stopping(10)])    
    y_pred = bst.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    
    mlflow.log_metric("val_rmse", float(rmse))
    mlflow.log_metric("val_r2", float(r2))
    mlflow.lightgbm.log_model(bst, "model")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

