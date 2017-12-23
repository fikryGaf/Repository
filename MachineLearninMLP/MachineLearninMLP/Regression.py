
from __future__ import print_function
import numpy as np
import math
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# Load the data set
data = np.loadtxt('nonlinear.data')

# Separate the input from the output
X = data[:, 0:-1]
Y = data[:, -1]
Y=(data[:, -1]+1)/2
N, d = X.shape
positive_class = X[Y==1., :]
negative_class = X[Y==0., :]
def sigmoid(x):
  return 1. / (1. + math.exp(-x))

def Compute_Error(w,b):
    nb_error=0
    # to ensure the minimum will be Y[i] * scale_product
    gamma=1000
    for i in range(N):
        scale_product = np.dot(X[i,:],w) + b
        gamma = min(gamma, Y[i] * scale_product)
        if sigmoid(scale_product) != Y[i]:
                nb_error+=1
    return nb_error/N, gamma

def Visualize(w,b):

    plt.plot(positive_class[:,0],positive_class[:,1],'ob')
    plt.plot(negative_class[:,0],negative_class[:,1],'xr')
    XX=np.linspace(0,1,100)
    if w[1] !=0 : 
        YY=-(w[0]*XX +b)/w[1]
        plt.plot(XX,YY,'.g')
        plt.show()
# Separate the positive from the negative class
'''
positive_class = X[Y==1., :]
negative_class = X[Y==-1., :]
plt.plot(positive_class[:,0],positive_class[:,1],'ob')
plt.plot(negative_class[:,0],negative_class[:,1],'or')
'''
#plt.plot(negative_class,'or')
'''
Wh=np.array([.45,.461])
bh=-.45
XXh=np.linspace(0,1,100)
YYh=-(Wh[0]*XXh +bh)/Wh[1]
print("HYPOTHISED: ",Wh,bh)
plt.plot(XXh,YYh,'.b')
'''

"""
w=np.zeros(2)
b=0
eta=
"""
W=np.zeros(2)
b=0
eta=.1

gamma= -1
epoch=1
while True:

    old_w=W.copy()
    for i in range(N):
        #Predicted value
        modelVal=np.dot(X[i,:],W) + b
        h_i= sigmoid(modelVal)
        # W learing rule
        W += eta * (Y[i] - h_i) * X[i, :]
        # b learing rule
        b += eta * (Y[i] - h_i)

    error, gamma=Compute_Error(W,b)
    print('Epoch:', epoch)
    print('W:',W,'B:',b)
    print('Error',error)
    print('Functional Margin',gamma)
    print('Weight variation',np.linalg.norm(W - old_w)/np.linalg.norm(old_w))
    print('______________________________________________')
    epoch+=1
    # 1% of the old w 16-3
    if np.linalg.norm(W - old_w) <.01 * np.linalg.norm(W):
        break
    '''
    if gamma >0:
        break
    '''
print('Norm of the weight vector:',np.linalg.norm(W))
Visualize(W,b)

 