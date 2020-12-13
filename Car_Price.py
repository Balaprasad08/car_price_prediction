import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('E:\\prasad\\practice\\Model Deploy\\Car Price Prediction')
df=pd.read_csv('car data.csv')
df.head(2)
df.shape
df.info()
df.isnull().sum()
df['Current_years']=2020
df.head(3)
df['No_of_years']=df['Current_years']-df['Year']
df.head()
df.drop(['Year','Current_years','Car_Name'],axis=1,inplace=True)
df.head()
df.shape
print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
df=pd.get_dummies(df,drop_first=True)
df.shape
df.head(2)
df.corr()
plt.figure(figsize=(16,9))
sns.heatmap(df.corr(),annot=True,cmap='Dark2_r',linewidths=0.3)
df.head(2)
X=df.iloc[:,1:]
X.head(2)
y=df.iloc[:,0]
y.head(2)
X['Owner'].unique()
from sklearn.model_selection import train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr.feature_importances_
feature_importances=pd.Series(etr.feature_importances_,index=X_train.columns)
feature_importances.nlargest(5).plot(kind='barh')
plt.show()
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
random_grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf}
print(random_grid)
rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,n_jobs=-1,cv=5,random_state=42,verbose=2)
rf_random.fit(X_train,y_train)
rf_random.best_estimator_
rf_random.best_index_
rf_random.best_params_
rf_random.best_score_
prediction=rf_random.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)
# ### Creat Model in Pickle & Joblib
import pickle
import joblib
pickle.dump(rf_random,open('car_model.pickle','wb'))
joblib.dump(rf_random,'car_model.joblib')
# #### Load Pickle Model
model_pkl=pickle.load(open('car_model.pickle','rb'))
y_pred=model_pkl.predict(X_test)
model_pkl.score(X_train,y_train)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
def check_model(Model,X_train,X_test,y_train,y_test):
    Model.fit(X_train,y_train)
    y_pred=Model.predict(X_test)
    print('r2_score:',r2_score(y_test,y_pred))
check_model(RandomForestRegressor(),X_train,X_test,y_train,y_test)
check_model(LinearRegression(),X_train,X_test,y_train,y_test)
check_model(SVR(),X_train,X_test,y_train,y_test)
check_model(DecisionTreeRegressor(),X_train,X_test,y_train,y_test)
check_model(KNeighborsRegressor(),X_train,X_test,y_train,y_test)
# #### Select RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('r2_score:',r2_score(y_test,y_pred))
rf.score(X_train,y_train)
rf.score(X_test,y_test)
X_test.shape
X_test.head(2)