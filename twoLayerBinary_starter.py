import numpy as np
from load_dataset import mnist
import matplotlib.pyplot as plt
import pdb

def tanh(Z):
    
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
   
    A = cache["Z"]
    A1 = 1-np.square(np.tanh(A))
    dZ = 1-A1
    return dZ

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    
    A = cache["Z"]
    A1,t = sigmoid(A)
    A2 = (1-A1)*(A1)
    dZ = dA*A2
    
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    
    np.random.seed(0)
    W1 = np.random.randn(n_h,n_in)*0.01                  #.random.randn
    b1 = np.random.randn(n_h,1)*0.01 
    W2 = np.random.randn(1,n_h)*0.01 
    b2 = np.random.randn(1,1)*0.01 

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):
   
    Z = np.dot(W,A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    
    m = Y.shape[1]
    cost = np.sum(Y*np.log10(A2) + (1-Y)*np.log10(1-A2))
    cost = cost*(-1/m)
    #cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache, W, b):
  
    A = cache["A"]
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis = 1 , keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
   
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def classify(X, parameters):
    
    activation = "sigmoid"
    A1,t1 = layer_forward(X, parameters["W1"], parameters["b1"], activation)
    A2,t2= layer_forward(A1, parameters["W2"], parameters["b2"], activation)
    YPred = []
    for a in np.nditer(A2):
        if a >= 0.5:
            YPred.append(1)
        else:
            YPred.append(0)
    
    YPred = np.matrix(YPred) 
    


    return YPred

def two_layer_network(X, Y, net_dims, num_iterations=2000, learning_rate=0.1):
    
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    activation = "sigmoid"
    A0 = X
    costs = []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, cache1 = layer_forward(A0,parameters["W1"],parameters["b1"],activation)
        A2,cache2 = layer_forward(A1,parameters["W2"],parameters["b2"],activation)
        
        # cost estimation
        ### CODE HERE
        cost=cost_estimate(A2,Y)
        # Backward Propagation
        ### CODE HERE
        m = A2.shape[1]
        dA = (Y/A2 - (Y-1)/(A2-1))
        dA = -(1/m)*dA

        dA_prev2, dW2, db2 = layer_backward(dA,cache2,parameters["W2"],parameters["b2"],activation)
        dA_prev1, dW1 , db1 = layer_backward(dA_prev2,cache1,parameters["W1"],parameters["b2"],activation)
        #update parameters
        ### CODE HERE
        parameters["W1"] = parameters["W1"] - learning_rate*dW1
        parameters["W2"] = parameters["W2"] - learning_rate*dW2
        parameters["b1"] = parameters["b1"] - learning_rate*db1
        parameters["b2"] = parameters["b2"] - learning_rate*db2

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    
    return costs, parameters

def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 2 and 3
    train_data, train_label, test_data, test_label = \
                mnist(ntrain=6000,ntest=1000,digit_range=[2,4])

    n_in, m = train_data.shape
    n_fin = 1
    n_h = 500
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 1000


    costs, parameters = two_layer_network(train_data, train_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)
    m1 = train_data.shape[1]
    m2 = test_data.shape[1]
    trAcc = 100-(np.sum(np.absolute(train_label - train_Pred))/m1)*100
    teAcc = 100-(np.sum(np.absolute(test_label - test_Pred))/m2)*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    # CODE HERE TO PLOT costs vs iterations
    points = np.arange(0,100)
    plt.plot(points,costs)
    plt.savefig("Error vs iterations")




if __name__ == "__main__":
    main()




