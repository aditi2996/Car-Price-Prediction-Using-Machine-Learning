#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Prediction
# 
# ## 1) Problem statement.
# 
# * This dataset comprises used cars sold on cardehko.com in India as well as important features of these cars.
# * If user can predict the price of the car based on input features.
# * Prediction results can be used to give new seller the price suggestion based on market condition.
# 
# ## 2) Data Collection.
# * The Dataset is collected from scrapping from cardheko webiste
# * The data consists of 13 column and 15411 rows.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("cardekho_imputated.csv", index_col=[0])


# In[3]:


df.head()


# ## Data Cleaning
# ### Handling Missing values

# * Handling Missing values 
# * Handling Duplicates
# * Check data type
# * Understand the dataset

# In[4]:


## Check Null Values
##Check features with nan value
df.isnull().sum()


# In[5]:


## Remove Unnecessary Columns
df.drop('car_name', axis=1, inplace=True)
df.drop('brand', axis=1, inplace=True)


# In[6]:


df.head()


# In[7]:


df['model'].unique()


# In[8]:


## Getting All Different Types OF Features
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print('Num of Numerical Features :', len(num_features))
cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print('Num of Categorical Features :', len(cat_features))
discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]
print('Num of Discrete Features :',len(discrete_features))
continuous_features=[feature for feature in num_features if feature not in discrete_features]
print('Num of Continuous Features :',len(continuous_features))


# In[9]:


## Indpendent and dependent features
from sklearn.model_selection import train_test_split
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']


# In[10]:


X.head()


# ## Feature Encoding and Scaling
# **One Hot Encoding for Columns which had lesser unique values and not ordinal**
# * One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.

# In[11]:


len(df['model'].unique())


# In[12]:


df['model'].value_counts()


# In[13]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X['model']=le.fit_transform(X['model'])


# In[14]:


X.head()


# In[15]:


len(df['seller_type'].unique()),len(df['fuel_type'].unique()),len(df['transmission_type'].unique())


# In[16]:


# Create Column Transformer with 3 types of transformers
num_features = X.select_dtypes(exclude="object").columns
onehot_columns = ['seller_type','fuel_type','transmission_type']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features)
        
    ],remainder='passthrough'
    
)


# In[17]:


X=preprocessor.fit_transform(X)


# In[18]:


pd.DataFrame(X)


# In[19]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# 
# 

# In[20]:


X_train


# ## Model Training And Model Selection

# In[21]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[22]:


##Create a Function to Evaluate Model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[23]:


## Beginning Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Adaboost Regressor":AdaBoostRegressor(),
    "Graident BoostRegressor":GradientBoostingRegressor(),
    "Xgboost Regressor":XGBRegressor()
   
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    adjusted_r2_test = 1 - (1 - model_test_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    adjusted_r2_train = 1 - (1 - model_train_r2) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)

    
    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    print("- Adjusted R2 Score: {:.4f}".format(adjusted_r2_train))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    print("- Adjusted R2 Score: {:.4f}".format(adjusted_r2_test))
    
    print('='*35)
    print('\n')


# In[24]:


#Initialize few parameter for Hyperparamter tuning

rf_params = {"max_depth": [5, 8, 15, None, 10,20],
             "max_features": [5, 7,8, "auto", "sqrt","log2"],
             "min_samples_split": [2, 5, 8, 10, 15, 20],
             "n_estimators": [100, 200, 500, 1000, 1500, 2000]}

xgboost_params = {"learning_rate": [0.01,0.05,0.1,0.2],
                  "max_depth": [5, 8, 12, 20, 30,None],
                  "n_estimators": [100, 200, 300,400,500],
                  "colsample_bytree": [0.3, 0.316, 0.5, 0.8, 1],
                  "subsample": [0.5, 0.7, 1],
                  "gamma": [0, 0.1, 0.5, 1],
                  "min_child_weight": [1, 3, 5, 10]}


# In[25]:


# Models list for Hyperparameter tuning
randomcv_models = [
                   ("RF", RandomForestRegressor(), rf_params),
                   ("XGboost",XGBRegressor(),xgboost_params)
                   
                   ]


# In[26]:


##Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


# In[30]:


## Retraining the models with best parameters
models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=500, min_samples_split=2, max_features=5, max_depth=15, 
                                                     n_jobs=-1),
     "Xgboost Regressor":XGBRegressor(n_estimators= 100,learning_rate=0.05,min_child_weight=1,
                                         max_depth=None,gamma= 0.5,colsample_bytree=0.5)
    
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    adjusted_r2_test = 1 - (1 - model_test_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    adjusted_r2_train = 1 - (1 - model_train_r2) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)

    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    print("- Adjusted R2 Score: {:.4f}".format(adjusted_r2_train))
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    print("- Adjusted R2 Score: {:.4f}".format(adjusted_r2_test))
    print('='*35)
    print('\n')


# In[ ]:




