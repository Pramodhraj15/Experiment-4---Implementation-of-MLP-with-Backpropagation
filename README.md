import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
irisdata=pd.read_csv("iris.txt")
x=irisdata.iloc[:,0:4]
y=irisdata.select_dtypes(include=[object])
le=preprocessing.LabelEncoder()
y=y.apply(le.fit_transform)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(x_train,y_train.values.ravel())
predictions=mlp.predict(x_test)
print(predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
