import numpy as np
import matplotlib.pyplot as plt

# Hours 
x = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])

# Pass 
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

w = 0
b = 0

def plot(w, b):
    y_predict_1 = np.dot(x, w) + b
    y_predict = sigmoid(y_predict_1)

    plt.plot(x, y, 'x')
    plt.plot(x, y_predict)

    plt.ylabel('Hours') 
    plt.xlabel('Pass or Fail') 

    plt.show()

def sigmoid(z):
    g = 1 / ( 1 + np.exp(-z) )

    return g
    
def cost_function(x, y, w, b):
    m = x.shape[0]

    y_predict_1 = np.dot(x, w) + b
    y_predict = sigmoid(y_predict_1)

    loss = 0

    loss = -y * np.log(y_predict) - (1 - y) * np.log(1 - y_predict)
    total_cost = (1 / m) * np.sum(loss)
    return total_cost

#print(cost_function(x,y,w,b))

def compute_gradient(x, y, w, b):
    m = x.shape[0]

    y_predict = sigmoid(np.dot(x, w) + b)

    W_Der = np.sum((y_predict - y) * x) / m
    B_Der = np.sum((y_predict - y)) / m

    return W_Der, B_Der

#print(compute_gradient(x, y, w, b))

def gradient_descent(x, y, w, b, iterations, alpha):
    m = x.shape[0]

    for i in range(iterations):
        W_Der, B_Der = compute_gradient(x, y, w, b)

        w = w - alpha * W_Der
        b = b - alpha * B_Der

        if i%1000 == 0:
            print(f"Iteration: {i}, W = {w}, B = {b}")

    return w, b

W, B = (gradient_descent(x, y, w, b, 300000, 0.01))

plot(W, B)