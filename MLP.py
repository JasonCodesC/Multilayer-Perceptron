import numpy as np
import pandas as pd

def main():
    
    #Reads in data from csv and wrangles the data so it is ready to be used
    DataFromCSV = pd.read_csv('train.csv')
    ArrayData = np.array(DataFromCSV)
    m, n = ArrayData.shape
    np.random.shuffle(ArrayData)
    
    #initializing variables for training and testing
    DataP = ArrayData[0:1000].T
    Ypart = DataP[0]
    Xpart = (DataP[1:n]) / 255
    TrainData = ArrayData[1000:m].T
    YpartT = TrainData[0]
    XpartT = (TrainData[1:n]) / 255

    #Calls the gradient descent function to get the correct weights and biases and assess accuracy
    Weight1, Bias1, Weight2, Bias2 = SGD(XpartT, YpartT, 130000, 0.021)
    print("Here are you first set of weights", Weight1)
    print("Here are you first set of Biases", Bias1)
    print("Here are you second set of weights", Weight2)
    print("Here are you second set of Biases", Bias2)


  
#Gradient Descent for minimizing the cost function and optimizing accuracy
def SGD(A, B, numT, Q) :
    Weight1, Bias1, Weight2, Bias2 = __init__vars()
    for i in range(numT):
        NW1, NB1, NW2, NB2 = forwardP(Weight1, Bias1, Weight2, Bias2, A)
        w1, b1, w2, b2 = backP(NW1, NB2, NW2, NB2, Weight2, A, B)
        b1 = b1.reshape(Bias1.shape)
        b2 = b2.reshape(Bias2.shape) 
        Weight1, Bias1, Weight2, Bias2 = update_params(Weight1, Bias1, Weight2, Bias2, w1, b1, w2, b2, Q)
        if i % 1000 == 0:
            print("Times: ", i)
            print("Accuracy: ", acc(predictions(NB2), B))
    return Weight1, Bias1, Weight2, Bias2

#Forward Propagation 
def forwardP(Weight1, Bias1, Weight2, Bias2, arr):
    NW1 = (Weight1.dot(arr)) + Bias1
    NB1 = ReLU(NW1)
    NW2 = Weight2.dot(NB1) + Bias2
    NB2 = SM(NB1)
    return NW1, NB1, NW2, NB2

#Back Propagation algorithm for changing weights and biases 
def backP(NW1, NB1, NW2, NB2, Weight2, A, B):
    m = B.size
    NY = SetOne(B)
    Z2 = NB2 - NY
    w2 = 1 / m * Z2.dot(NB1.T)
    B2 = 1 / m * np.sum(Z2, axis=1, keepdims=True)
    Z1 = Weight2.T.dot(Z2) * ActivationPrime(NW1)
    w1 = 1 / m * Z2.dot(A.T)
    B1 = 1 / m * np.sum(Z1, axis=1, keepdims=True)
    return w1, B1, w2, B2

#Updates paramaters using the learning rate
def update_params(Weight1, Bias1, Weight2, Bias2, w1, B1, w2, B2, Q):
    Weight1 = Weight1 - Q * w1
    Bias1 = Bias1 - (Q * B1.reshape(Bias1.shape) )
    Weight2 = Weight2 - Q * w2
    Bias2 = Bias2 - Q * B2
    return Weight1, Bias1, Weight2, Bias2


#Gives weights and biases random values so they train better
def __init__vars():
    Weight1 = np.random.rand(10,784) - 0.5
    Bias1 = np.random.rand(10, 1) - 0.5
    Weight2 = np.random.rand(10,10) - 0.5
    Bias2 = np.random.rand(10, 1) - 0.5
    return Weight1, Bias1, Weight2, Bias2

#ReLU function activing as the activation function for 0 =< nums <= 1
def ReLU(N):
    return np.maximum(0, N)
    
#Partial Derivative of ReLU function
def ActivationPrime(N):
    return N > 0

#Normalizes the output of a network to a probability distribution
def SM(N):
    expN = np.exp(N - np.max(N, axis=0)) 
    return expN / np.sum(expN, axis=0)
    

    
#Converts to a "One-Hot Array"
def SetOne(Y) :
    NY = np.zeros((Y.size, Y.max() + 1))
    NY[np.arange(Y.size), Y] = 1
    return NY.T



#Functions to provide accuracy while training
def predictions(NB2):
    return np.argmax(NB2, 0)
def acc(predictions, B):
    print(predictions, B)
    return np.sum(predictions == B) / B.size


main()
