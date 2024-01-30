import pickle

with open('Iris_ML_Logistic','rb') as f:
    model = pickle.load(f)
predicted  = model.predict([[5.7, 2.6, 3.5, 1. ]])
if predicted ==1:
    print("VersiColour")
elif predicted == 0:
    print("Setosa")
else:
    print("Virginica")