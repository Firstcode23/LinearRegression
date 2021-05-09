import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
#import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



# read data from file
def read_data():
    data = pd.read_csv('C:\\Sudarshan\\ds_training\\Python_HackerRank\\Datasets\\Big Mart Sales Prediction\\train.csv')
    return data

#visualization of data
def data_visualization(data):
    print(data.info())
    print(data.shape)
    print(data.isna().sum())
# Imputation
def missingdata_imputation(data):
    data.Item_Weight.fillna(data.Item_Weight.mean(),inplace = True)
    data.Outlet_Size.fillna(data.Outlet_Size.mode()[0],inplace = True)
    print(data.isna().sum())
    return data

def label_encoder(data):
    cat_col = ['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
    le = LabelEncoder()
    data[cat_col] = data[cat_col].apply(lambda col:le.fit_transform(col))
    return data
    #print(data.head(5))
    #print(data.info())

def df_corelation(data):
    corr_col = ['Item_Weight','Item_Visibility','Item_MRP','Item_Outlet_Sales']
    df_corr = data[corr_col]
    #print(df_corr.head())
    plt.figure(figsize=(12,10))
    cor = df_corr.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

def df_anova(data):
    cat_col = ['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
    x_anova = data[cat_col]
    y_anova = data['Item_Outlet_Sales']
    fs_anova = SelectKBest(score_func= f_classif, k='all')
    fs_anova.fit(x_anova,y_anova)
    for i in range(len(fs_anova.scores_)):
        print(fs_anova.scores_[i])

def svm_regression(data):
    svm_cal = SVR(C=1.0,gamma='auto' ,kernel = 'linear')
    X = data.drop(['Item_Identifier','Item_Visibility','Item_Outlet_Sales'],axis = 1)
    y = data['Item_Outlet_Sales']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
    svm_cal.fit(X_train,y_train)
    y_predict = svm_cal.predict(X_test)
    #print('Prececison and Recall Score is :' ,precision_recall_fscore_support(y_test,y_predict,average = 'micro'))
    r_score =r2_score(y_test, y_predict)
    print(r_score)
    #print(svm_cal.score(X_train,y_train))
    #print(svm_cal.score(X_test,y_test))

def decesion_regressor(data,md):
    dt_regression = DecisionTreeRegressor(max_depth = md, min_samples_leaf= 100)
    X = data.drop(['Item_Identifier','Item_Visibility','Item_Outlet_Sales'],axis = 1)
    y = data['Item_Outlet_Sales']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
    dt_regression.fit(X_train,y_train)
    y_dt_predict = dt_regression.predict(X_test)
    r_dt_score =r2_score(y_test, y_dt_predict)
    print(r_dt_score)

def random_regressor(data,n_estimator,md):
    #parameters_grid =  {'bootstrap' : [True], 'max_depth': [5,10,None],'n_estimators':[10,20,30]}
    rf_regression = RandomForestRegressor(n_estimators= n_estimator, max_depth=md)
    #rf_regression = RandomForestRegressor(random_state = 42)
    #grid_search = GridSearchCV(estimator = rf_regression,param_grid = parameters_grid,cv=3,n_jobs = 1,verbose = 0,return_train_score =  True)
    X = data.drop(['Item_Identifier','Item_Visibility','Item_Outlet_Sales'],axis = 1)
    y = data['Item_Outlet_Sales']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
    rf_regression.fit(X_train,y_train)
    #grid_search.fit(X_train,y_train)
    #print(grid_search.best_params_)
    #print(best_grid.score(X_test, y_test))
    y_rf_regression_predict = rf_regression.predict(X_test)
    rf_regression_r_score = r2_score(y_test,y_rf_regression_predict)
    print(rf_regression_r_score)




training_data = read_data()
data_visualization(training_data)
im_data = missingdata_imputation(training_data)
label_data= label_encoder(im_data)
df_corelation(im_data)
df_anova(label_data)
svm_regression(label_data)
decesion_regressor(label_data,10)
random_regressor(label_data,30,5)




