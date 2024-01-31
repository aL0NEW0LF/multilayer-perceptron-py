from math import inf, sqrt
import sys
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 

class MyMlp(BaseEstimator, ClassifierMixin): 
    def __init__(self, params={'InputLayer': 4, 'HiddenLayerSizes': [3, 3, 3], 'HiddenLayerActivations': ['sigmoid', 'sigmoid', 'sigmoid'], 'OutputLayer': 3, 'LearningRate': 0.005, 'Epochs': 600, 'BiasHiddenValue': -1, 'BiasOutputValue': -1, 'OutputActivationFunction': 'sigmoid', 'ClassNumber': 3}):     
        self.inputLayer = params['InputLayer']
        self.hiddenLayerSizes = params['HiddenLayerSizes']
        self.OutputLayer = params['OutputLayer']
        self.learningRate = params['LearningRate']
        self.max_epochs = params['Epochs']
        self.BiasHiddenValue = params['BiasHiddenValue']
        self.BiasOutputValue = params['BiasOutputValue']
        self.activation = [self.activationFunctions[func_name] for func_name in params['HiddenLayerActivations']]
        self.derivative = [self.activationFunctions[func_name] for func_name in params['HiddenLayerActivations']]
        self.outputActivation = self.activationFunctions[params['OutputActivationFunction']]
        self.outputDerivative = self.activationFunctions[params['OutputActivationFunction']]
        
        self.WEIGHT_hidden = []

        'Starting Bias and Weights'
        for index, layerSize in enumerate(self.hiddenLayerSizes):
            if index == 0:
                if self.activation[index] == self.activationFunctions['sigmoid'] or self.activation[index] == self.activationFunctions['tanh'] or self.activation[index] == self.activationFunctions['softmax']:
                    self.WEIGHT_hidden.append(self.starting_weights_sig_tanh(layerSize, self.inputLayer))
                else:
                    self.WEIGHT_hidden.append(self.starting_weights_relu(layerSize, self.inputLayer))
            else:
                if self.activation[index] == self.activationFunctions['sigmoid'] or self.activation[index] == self.activationFunctions['tanh'] or self.activation[index] == self.activationFunctions['softmax']:
                    self.WEIGHT_hidden.append(self.starting_weights_sig_tanh(layerSize, self.hiddenLayerSizes[index-1]))
                else:
                    self.WEIGHT_hidden.append(self.starting_weights_relu(layerSize, self.hiddenLayerSizes[index-1]))

        if self.outputActivation == self.activationFunctions['sigmoid'] or self.outputActivation == self.activationFunctions['tanh'] or self.activation[index] == self.activationFunctions['softmax']:
            self.WEIGHT_output = self.starting_weights_sig_tanh(self.OutputLayer, self.hiddenLayerSizes[-1])
        else:
            self.WEIGHT_output = self.starting_weights_relu(self.OutputLayer, self.hiddenLayerSizes[-1])

        self.BIAS_hidden = [[self.BiasHiddenValue for i in range(self.hiddenLayerSizes[j])] for j in range(len(self.hiddenLayerSizes))]
        self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])
        self.classes_number = params['ClassNumber']

    def starting_weights_sig_tanh(self, x, y):
        'Returns a matrix of weights with random values'
        lower, upper = -(1.0 / sqrt(x)), (1.0 / sqrt(x))
        initWeights = []
        for i in range(y):
            weightsList = []
            for j in range(x):
                random.seed(random.randint(0, x))
                weightsList.append(random.uniform(lower, upper))
            initWeights.append(weightsList)
        return np.array(initWeights)
    
    def starting_weights_relu(self, x, y):
        'Returns a matrix of weights with random values'
        sigma = sqrt(2.0 / (x))
        initWeights = []
        for i in range(y):
            weightsList = []
            for j in range(x):
                random.seed(random.randint(0, x))
                weightsList.append(random.gauss(0, sigma))
            initWeights.append(weightsList)
        return np.array(initWeights)

    activationFunctions = {
            'sigmoid': (lambda x: 1/(1 + np.exp(-x)) if x.all() >= 0 else np.exp(x)/(1 + np.exp(x))),
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
 
    def clip_gradients(self, gradients, max_value):
        total_norm = np.sqrt(np.sum([np.sum(np.square(grad)) for grad in gradients]))
        clip_coef = max_value / (total_norm + 1e-6)  # add small epsilon to prevent division by zero
        if clip_coef < 1:
            return [clip_coef * grad for grad in gradients]
        return gradients
    
    def back_propagation(self, x):
        DELTA_output = []
        'Error: OutputLayer'
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1)*(ERROR_output) * self.outputDerivative(self.OUTPUT_L2))
        
        arrayStore = []
        'Update weights OutputLayer and HiddenLayer'
        for j in range(self.OutputLayer):
            for i in range(self.hiddenLayerSizes[-1]):
                self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[-1][i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])

        'Error: HiddenLayer' 
        DELTA_hidden = [np.matmul(self.WEIGHT_output[i], DELTA_output) * self.derivative[i](self.OUTPUT_L1[i]) for i in range(len(self.hiddenLayerSizes))]
        
        DELTA_hidden = self.clip_gradients(DELTA_hidden, 1)

        'Update weights HiddenLayer and InputLayer(x)'
        for j, layerSize in reversed(list(enumerate(self.hiddenLayerSizes))):
            """ print('j: ', j)
            print('layerSize: ', layerSize) """
            for k in range(layerSize):
                """ print('k: ', k) """
                for i in range(self.hiddenLayerSizes[j-1] if j > 0 else self.inputLayer):
                    """ print('i: ', i)
                    print('Before-----------------------------------------------------------------')
                    print('self.WEIGHT_hidden: ', self.WEIGHT_hidden)
                    print('self.WEIGHT_hidden length: ', len(self.WEIGHT_hidden))
                    print('self.BIAS_hidden: ', self.BIAS_hidden)
                    print('self.BIAS_hidden length: ', len(self.BIAS_hidden)) """
                    self.WEIGHT_hidden[j][i][k] -= (self.learningRate * (DELTA_hidden[j][k]))
                    self.BIAS_hidden[j][k] -= (self.learningRate * DELTA_hidden[j][k])
                    """ print('After------------------------------------------------------------------')
                    print('self.WEIGHT_hidden: ', self.WEIGHT_hidden)
                    print('self.WEIGHT_hidden length: ', len(self.WEIGHT_hidden))
                    print('self.BIAS_hidden: ', self.BIAS_hidden)
                    print('self.BIAS_hidden length: ', len(self.BIAS_hidden)) """

    def predict(self, X, y):
        'Returns the predictions for every element of X'
        my_predictions = []
        'Forward Propagation'
        forward = []
        for i in range(len(self.hiddenLayerSizes)):
            if i == 0:
                forward.append(np.matmul(X, self.WEIGHT_hidden[i]) + self.BIAS_hidden[i])
            else:
                forward.append(np.matmul(forward[i-1], self.WEIGHT_hidden[i]) + self.BIAS_hidden[i])

        forward.append(np.matmul(forward[-1], self.WEIGHT_output) + self.BIAS_output)

        print('forward[-1]: ', forward[-1])       

        for i in forward[-1]:
            my_predictions.append(max(enumerate(i), key=lambda x:x[1])[0])
            
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
                '(Forward Propagation)'
                self.OUTPUT_L1 = []
                for i in range(len(self.hiddenLayerSizes)):
                    if i == 0:
                        self.OUTPUT_L1.append(self.activation[i](np.dot(inputs, self.WEIGHT_hidden[i]) + self.BIAS_hidden[i]))
                    else:
                        self.OUTPUT_L1.append(self.activation[i](np.dot(self.OUTPUT_L1[i-1],self.WEIGHT_hidden[i]) + self.BIAS_hidden[i]))

                self.OUTPUT_L2 = self.outputActivation(np.dot(self.OUTPUT_L1[-1], self.WEIGHT_output) + self.BIAS_output)

                'One-Hot-Encoding'
                self.output = np.array(encoder.transform(y[idx].reshape(-1,1)).toarray()).flatten()

                square_error = 0
                for i in range(self.OutputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i])**2
                    square_error = (square_error + (0.05 * erro))
                    total_error = total_error + square_error
                
                'Backpropagation : Update Weights'
                self.back_propagation(inputs)
            
            total_error = (total_error / n)
            if((count_epoch % 50 == 0)or(count_epoch == 1)):
                print("Epoch ", count_epoch, "- Total Error: ",total_error)
                error_array.append(total_error)
                epoch_array.append(count_epoch)
                
            W0.append(list(self.WEIGHT_hidden))
            W1.append(list(self.WEIGHT_output))
             
            count_epoch += 1

        print('Model fitted.')
        """ print('W0: ', W0)
        print('W1: ', W1) """
        self.show_err_graphic(error_array,epoch_array)
        
        """ plt.plot(W0[0])
        plt.title('Weight Hidden update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3', 'neuron4', 'neuron5'])
        plt.ylabel('Value Weight')
        plt.show()
        
        plt.plot(W1[0])
        plt.title('Weight Output update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3'])
        plt.ylabel('Value Weight')
        plt.show() """

        return self
    
    def show_err_graphic(self,v_erro,v_epoca):
        plt.figure(figsize=(9,4))
        plt.plot(v_epoca, v_erro, "m-",color="b", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Squared error (MSE) ")
        plt.title("Error Minimization")
        plt.show()