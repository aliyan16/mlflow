import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')

wine=load_wine()
x=wine.data
y=wine.target


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
max_depth=4
n_estimators=10

rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)
accuracy=accuracy_score(ypred,ytest)
confusionMat=confusion_matrix(ypred,ytest)
plt.figure(figsize=(6,6))
sns.heatmap(confusionMat,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('ConfusionMat.png')

with mlflow.start_run():
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max depth',max_depth)
    mlflow.log_param('n estimators',n_estimators)
    mlflow.log_artifact('confusionMat.png')
    mlflow.log_artifact(__file__)

