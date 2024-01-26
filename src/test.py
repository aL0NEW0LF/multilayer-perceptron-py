import numpy as np
from mlp import MultiLayerPerceptron
from mlp2 import Mlp
from mymlp import MyMlp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split
import pandas as pd

""" iris = datasets.load_iris()

X = iris.data
y = iris.target

y """

titanic = pd.read_excel("D:/Projects/multilayer-perceptron-py/src/titanic_clean.xlsx", header=0)

X = titanic.drop("survived", axis=1).to_numpy()
y = titanic["survived"].to_numpy()

dictionary2 = {'InputLayer':3, 
               'HiddenLayerSizes': [8], 
               'HiddenLayerActivations': ['sigmoid'], 
               'OutputLayer':2, 
               'Epochs':700, 
               'LearningRate':0.005,
               'BiasHiddenValue':-1, 
               'BiasOutputValue':-1, 
               'OutputActivationFunction':'sigmoid', 
               'ClassNumber':2}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print(y_train)

Perceptron2 = MyMlp(dictionary2)

""" print(Perceptron2.inputLayer)
print(Perceptron2.hiddenLayerSizes)
for i in Perceptron2.activation:
    print(i)
print(Perceptron2.activation)
print(Perceptron2.derivative)w
print(Perceptron2.OutputLayer)
print(Perceptron2.max_epochs)
print(Perceptron2.learningRate)
print(Perceptron2.BiasHiddenValue)
print(Perceptron2.BiasOutputValue)
print(Perceptron2.outputActivation)
print(Perceptron2.outputDerivative)
print(Perceptron2.classes_number)
print(Perceptron2.WEIGHT_hidden)
print(len(Perceptron2.WEIGHT_hidden)) 
for i, j in enumerate(Perceptron2.WEIGHT_hidden):
    print(i)
    print(pd.DataFrame(j))   
print(Perceptron2.WEIGHT_output)
print(len(Perceptron2.WEIGHT_output))
print(Perceptron2.BIAS_hidden)
print(Perceptron2.BIAS_output) """

""" Perceptron2.fit(X_train, y_train)

predictions, df = Perceptron2.predict(X_test, y_test)

print(predictions)

print(Perceptron2.accuracy_score(df['hoped_output'], df['output'])) """

dictionary = {'InputLayer':3, 'HiddenLayer':8, 'OutputLayer':2, 'Epochs':700, 'LearningRate':0.005,'BiasHiddenValue':-1, 'BiasOutputValue':-1, 'ActivationFunction':'sigmoid', 'ClassNumber':2}

Perceptron = MultiLayerPerceptron(dictionary)
Perceptron.fit(X_train, y_train)

preds, hits = Perceptron.predict(X_test, y_test)

print(preds)

print(Perceptron.accuracy_score(hits['hoped_output'], hits['output']))