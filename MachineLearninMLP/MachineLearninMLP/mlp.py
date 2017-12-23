from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# fikry

# Load the data set
data = np.loadtxt('nonlinear_classification.data')

# Separate the input from the output
X = data[:, :2]
T = data[:, 2]
N, d = X.shape

# Parameters
eta = 0.05 # learning rate
K = 15 # Number of hidden neurons

# Weights and biases
max_val = 1.0
W_hid  = np.random.uniform(-max_val, max_val, (d, K)) 
b_hid  = np.random.uniform(-max_val, max_val, K)
W_out = np.random.uniform(-max_val, max_val, K) 
b_out = np.random.uniform(-max_val, max_val, 1)

# Logistic transfer function for the hidden neurons 
def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

# Threshold transfer function for the output neuron
def threshold(x):
    data = x.copy()
    data[data > 0.] = 1.
    data[data < 0.] = -1.
    return data

def feedforward(x, W_hid, b_hid, W_out, b_out):
    # Hidden layer
    h = logistic(np.dot(x, W_hid) + b_hid)
    # Output layer
    y = threshold(np.dot(h, W_out) + b_out)  
    return h, y

def plot_classification(X, T, W_hid, b_hid, W_out, b_out):
    
    # True values
    positive_class = X[T>0]
    negative_class = X[T<0]

    # Prediction
    H, Y = feedforward(X, W_hid, b_hid, W_out, b_out) 
    misclassification = X[Y!=T]

    # Plot
    plt.plot(positive_class[:,0], positive_class[:,1], 'bo')
    plt.plot(negative_class[:,0], negative_class[:,1], 'go')
    plt.plot(misclassification[:,0], misclassification[:,1], 'ro')
    plt.show()

# Perform maximally 2000 epochs
errors = []
for epoch in range(2000):
    nb_errors = 0
    for i in range(N):
        # Input vector
        x = X[i, :]

        # Desired output
        t = T[i]
        
        # Feedforward pass
        h, y = feedforward(x, W_hid, b_hid, W_out, b_out) 

        # Count the number of misclassifications
        if t != y: 
            nb_errors += 1

        # Output error
        delta_out = 0.0 # TODO

        # Hidden error
        delta_hidden = 0.0 # TODO
        
        # Learn the output weights
        W_out += 0.0 # TODO

        # Learn the output bias
        b_out += 0.0 # TODO

        # Learn the hidden weights
        for j in range(K):
            W_hid[:, j] += 0.0 # TODO

        # Learn the hidden biases
        b_hid += 0.0 # TODO

    # Compute the error rate
    print('Error rate', nb_errors/float(N))
    errors.append(nb_errors/float(N))
    
    # Stop when the error rate is 0
    if nb_errors == 0:
        break


print('Number of epochs needed:', epoch+1)
plot_classification(X, T, W_hid, b_hid, W_out, b_out) 
plt.plot(errors)
plt.show()


