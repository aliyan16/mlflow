from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
data=load_breast_cancer()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
mlflow.autolog()
mlflow.set_experiment('Breast Cancer')
rf=RandomForestClassifier(random_state=42)
param_grid={
    'n_estimators':[10,20,50],
    'max_depth':[None,10,15,30]
}

grid=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
grid.fit(xtrain,ytrain)
bestParams=grid.best_params_
bestScore=grid.best_score_


with mlflow.start_run() as parent:
    for i in range(len(grid.cv_results_['params'])):
        with mlflow.start_run(nested=True)
        mlflow.log_params(grid.cv_results_['params'][i])
        mlflow.log_metric('accuracy',grid.cv_results_['mean_test_score'][i])
    mlflow.log_metric('accuracy',bestScore)
    trainDf=xtrain.copy()
    trainDf['target']=ytrain
    trainDf=mlflow.data.from_pandas(trainDf)
    mlflow.log_input(trainDf,'training')
    testDf=xtest.copy()
    testDf['target']=ytest
    testDf=mlflow.data.from_pandas(testDf)
    mlflow.log_input(testDf,'testing')
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(grid.best_estimator_,'Random Forest')
    mlflow.set_tag('Author','Aliyan')
