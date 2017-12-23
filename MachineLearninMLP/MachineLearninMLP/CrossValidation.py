from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
 
# Load the data set
data = np.loadtxt('polynome.data')
X = data[:, 0]
Y = data[:, 1]
N = len(X)

def visualize(w):
    # Plot the data
    plt.plot(X, Y, 'r.')
    # Plot the fitted curve 
    x = np.linspace(0., 1., 100)
    y = np.polyval(w, x)
    plt.plot(x, y, 'g-')
    plt.title('Polynomial regression with order ' + str(len(w)-1))
    plt.show()

# Apply polynomial regression of order 2 on the data
w = np.polyfit(X, Y, 2)

for deg in range(1,11):
    w=np.polyfit(X,Y,deg)
    predit=np.polyval(w,X)
    error=np.sum((Y-predit)**2)/(2*N)
    print('Polynomial of order',deg,'Training Error:',error)
    visualize(w)

# Crossvalidation
'''
for deg in range(1,11):
    X_train=data[: 11]
    X_test=data[11 :]
    Y_train=data[: 11]
    Y_test=data[11 :]

    w=np.polyfit(X_train,Y_train,deg)
    predit=np.polyval(w,X_test)
    error=np.sum((Y_test-predit)**2)/(2*N)
    print('Polynomial of order',deg,'Training Error:',error)
    visualize(w)
'''
# K-Fold
print('K-fold cross validation')
k=1
np_split=int(N/k)
X_sets=np.hsplit(X,np_split)
Y_sets=np.hsplit(Y,np_split)
for deg in range(1,11):
    training_Error=0.0
    test_Error=0.0
    for s in range(np_split):
        X_train=np.hstack((X_sets[i] for i in range(np_split) if not i==s))
        Y_train=np.hstack((Y_sets[i] for i in range(np_split) if not i==s))
        w=np.polyfit(X_train,Y_train,deg)
        Y_fit_train=np.polyval(w,X_train)
        Y_fit_test=np.polyval(w,X_sets[s])
        training_Error +=0.5 *np.dot((Y_train-Y_fit_train).T, (Y_train-Y_fit_train))/float(np_split)
        test_Error +=0.5 *np.dot((Y_sets[s]-Y_fit_train).T, (Y_sets[s]-Y_fit_test))/float(np_split)
    print('Polynomial of order',deg,'Training Error:',train_error,'Test error',test_Error)
    # visualize(w)


