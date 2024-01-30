from sklearn.datasets import load_iris
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
Y = iris.target
df=pd.DataFrame(X,Y)
df.columns = iris.feature_names
feature_name = iris.feature_names
target_name = iris.target_names

xtr,xte,ytr,yte = train_test_split(X,Y,test_size=0.2)



lr = LogisticRegression()

lr.fit(xtr,ytr)


predicted = lr.predict([[6.3, 2.5, 4.9, 1.5]])

newdf = pd.DataFrame(predicted,columns=['Predicted_No'])
newdf['Name'] = list(predicted)
newdf['Name'].replace({0:"Setosa",1:"Versi",2:"Vergi"},inplace=True)

print(newdf)
print(lr.score(xte,yte))

#with open("Iris_ML_Logistic","wb") as f:
 #   pickle.dump(lr,f)



