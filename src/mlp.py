from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 

class MultiLayerPerceptron(BaseEstimator, ClassifierMixin): 
    def __init__(self, params={'InputLayer': 4, 'HiddenLayer': 3, 'OutputLayer': 3, 'LearningRate': 0.005, 'Epochs': 600, 'BiasHiddenValue': -1, 'BiasOutputValue': -1, 'ActivationFunction': 'sigmoid', 'ClassNumber': 3}):     
        self.inputLayer = params['InputLayer']
        self.hiddenLayer = params['HiddenLayer']
        self.OutputLayer = params['OutputLayer']
        self.learningRate = params['LearningRate']
        self.max_epochs = params['Epochs']
        self.BiasHiddenValue = params['BiasHiddenValue']
        self.BiasOutputValue = params['BiasOutputValue']
        self.activation = self.activationFunctions[params['ActivationFunction']]
        self.derivative = self.derivatives[params['ActivationFunction']]
        
        'Starting Bias and Weights'
        self.WEIGHT_hidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
        self.WEIGHT_output = self.starting_weights(self.OutputLayer, self.hiddenLayer)
        self.BIAS_hidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])
        self.classes_number = params['ClassNumber']
            
    def starting_weights(self, x, y):
        return [[2  * random.random() - 1 for i in range(x)] for j in range(y)]

    activationFunctions = {
            'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
            'tanh': (lambda x: np.tanh(x)),
            'Relu': (lambda x: x*(x > 0)),
            'softmax': (lambda x: np.exp(x) / np.sum(np.exp(x), axis=0))
            }
    derivatives = {
            'sigmoid': (lambda x: x*(1-x)),
            'tanh': (lambda x: 1-(np.tanh(x))**2),
            'Relu': (lambda x: 1 * (x>0)),
            'softmax': (lambda x: x*(1-x))
            }
 
    def Backpropagation_Algorithm(self, x):
        DELTA_output = []
        'Stage 1 - Error: OutputLayer'
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1)*(ERROR_output) * self.derivative(self.OUTPUT_L2))
        
        arrayStore = []
        'Stage 2 - Update weights OutputLayer and HiddenLayer'
        for i in range(self.hiddenLayer):
            for j in range(self.OutputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])
      
        'Stage 3 - Error: HiddenLayer'
        delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output) * self.derivative(self.OUTPUT_L1)
 
        'Stage 4 - Update weights HiddenLayer and InputLayer(x)'
        for i in range(self.OutputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate * (delta_hidden[j] * x[i]))
                self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j])
                
    def show_err_graphic(self,v_erro,v_epoca):
        plt.figure(figsize=(9,4))
        plt.plot(v_epoca, v_erro, "m-",color="b", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Squared error (MSE) ")
        plt.title("Error Minimization")
        plt.show()

    def predict(self, X, y):
        'Returns the predictions for every element of X'
        my_predictions = []
        'Forward Propagation'
        forward = np.matmul(X, self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output

        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x:x[1])[0])

        print('forward: ', forward) 
        print('forward[-1]: ', forward[-1])

        array_score = []
        for i in range(len(my_predictions)):
            array_score.append([i, my_predictions[i], y[i]])

                    
        dataframe = pd.DataFrame(array_score, columns=['_id', 'output', 'hoped_output'])
        return my_predictions, dataframe

    def accuracy_score(self, y, y_pred):
        'Returns the accuracy between the true labels and the predictions'
        return np.round(np.sum(y == y_pred) / len(y), 4)


    def fit(self, X, y):  
        count_epoch = 1
        total_error = 0
        n = len(X); 
        epoch_array = []
        error_array = []
        W0 = []
        W1 = []
        encoder = OneHotEncoder()
        encoder.fit(y.reshape(-1,1))
        while(count_epoch <= self.max_epochs):
            for idx, inputs in enumerate(X): 
                self.output = np.zeros(self.classes_number)
                'Stage 1 - (Forward Propagation)'
                self.OUTPUT_L1 = self.activation((np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T))
                self.OUTPUT_L2 = self.activation((np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T))
                'Stage 2 - One-Hot-Encoding'
                self.output = np.array(encoder.transform(y[idx].reshape(-1,1)).toarray()).flatten()

                square_error = 0
                for i in range(self.OutputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i])**2
                    square_error = (square_error + (0.05 * erro))
                    total_error = total_error + square_error
        
                'Backpropagation : Update Weights'
                self.Backpropagation_Algorithm(inputs)
            
            total_error = (total_error / n)
            if((count_epoch % 50 == 0)or(count_epoch == 1)):
                print("Epoch ", count_epoch, "- Total Error: ",total_error)
                error_array.append(total_error)
                epoch_array.append(count_epoch)
                
            W0.append(self.WEIGHT_hidden)
            W1.append(self.WEIGHT_output)
             
            count_epoch += 1

        print('Model fitted.')

        self.show_err_graphic(error_array,epoch_array)
        
        print('W0: ', W0)
        print('W1: ', W1)

        plt.plot(W0[0])
        plt.title('Weight Hidden update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3', 'neuron4', 'neuron5'])
        plt.ylabel('Value Weight')
        plt.show()
        
        plt.plot(W1[0])
        plt.title('Weight Output update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3'])
        plt.ylabel('Value Weight')
        plt.show()

        return self