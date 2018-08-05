
import numpy as np
from load_dataset import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
from sklearn.utils.extmath import softmax

def relu(Z):
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    
    A = softmax(Z)
    m = Y.shape[1]
    loss = 0 
    for i in range(A.shape[1]):
        loss = loss+np.log(A[int(Y[0][i])][i])

    loss = (-1/m)*loss
    cache = {}
    cache["A"] = A
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    
    A = cache["A"]
    m = Y.shape[1]
    one_hot = np.zeros([10,m])
    for i in range(m):
        one_hot[int(Y[0][i])][i]=1

    dZ = A-one_hot

    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1],net_dims[l])
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1],1)
    return parameters

def linear_forward(A, W, b):
    
    Z = np.dot(W,A) + b 
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    A_prev = cache["A"]
    
    dW = np.dot(dZ, A_prev.T)
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
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
        alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
     


    
    for l in range(L-1):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*gradients["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*gradients["db"+str(l+1)]

    alpha = decay_rate

    return parameters, alpha

def multi_layer_network(X, Y, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01):
    
    parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    costs = []
    for ii in range(num_iterations):
        A1, cache1 = multi_layer_forward(A0,parameters)
        A2 ,cache2, cost = softmax_cross_entropy_loss(A1,Y)


        dZ = softmax_cross_entropy_loss_der(Y,cache2)
        gradients = multi_layer_backward(dZ,cache1,parameters)

        parameters,alpha = update_parameters(parameters,gradients,ii,learning_rate,decay_rate)
        if ii % 10 == 0:
            costs.append(cost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
    
    return costs, parameters

def main():
   
    net_dims = ast.literal_eval( sys.argv[1] )
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    
    train_data, train_label, test_data, test_label = \
            mnist(ntrain=6000,ntest=1000,digit_range=[0,10])
    
    learning_rate = 0.2
    num_iterations = 500
    costs, parameters = multi_layer_network(train_data, train_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate)
    
    

    trAcc = None
    teAcc = None
    

if __name__ == "__main__":
    main()
