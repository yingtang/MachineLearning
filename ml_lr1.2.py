'''
Machine Learning on Coursera (Andrew Ng)
Exercise 1.2: Linear regression with multiple features

data set needed: 'ex1data2.txt'

@yingtang 
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def Featurenorm(X):
    global mu
    global sigma
    m,n = X.shape
    xnorm = np.ones((m,n))
    
    mu=[]
    sigma=[]
    
    '''note that the first column of X are all ones'''
    for i in range(1, n):
        mu.append(np.mean(X[:,i]))
        sigma.append(np.std(X[:,i]))
        xnorm[:,i]=(X[:,i]-mu[i-1])/sigma[i-1]

    return xnorm

def h(X, Theta):
    '''define hypothesis h'''
    return np.dot(X, Theta)

def CostFunc(X,Y,Theta):
    '''cost function'''
    m = Y.size
    diff = square_error = (h(X, Theta)-Y)
    J = 0.5*np.dot(diff.T, diff)/float(m)
    return J

def Gradient_descent(X,Y,Theta):
    '''Gradient Desent Methods'''
    global cost_history
    cost_history=[]
    m = Y.size
    #converge, used to make sure the gradient descent has reached a minimum 
    converge=100
   
    cost= CostFunc(X, Y, Theta)
    while converge > 1E-10:
        oldCost=cost
        cost_history.append(cost.flatten())
        #update feature parameter, Theta  
        Theta += alpha*np.dot(X.T, (Y-h(X, Theta)))/float(m)
        cost = CostFunc(X,Y,Theta)
        converge = abs(cost - oldCost)
    
    #return predictions
    return np.array(h(X, Theta))

'''Visualize original data'''
data=np.loadtxt('ex1data2.txt',delimiter=',')
fig1=plt.figure(1)

ax=fig1.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
ax.set_xlabel('size')
ax.set_ylabel('number of rooms')
ax.set_zlabel('price')
#########################################################

m,n=data.shape #dimensions of the input data
fit=np.ones(m, float) #fit is an array of predicted values

#add zeroth feature +1
Theta=np.ones((n,1), float) #theta
X=np.ones((m,n),float) #features X, dimension: m x 3 
Y=np.ones(shape=(m,1))  #y, dimension: m x 1

X[:,1:3]=data[:,0:2] #x is a m X 3 array
Y[:,0]=data[:,-1] #Y is a m x 1 array

#Learning rate
alpha=0.01
###########################################################

#normalize the features
normX=Featurenorm(X)

#Find the correct fitting parameter and give the prediction (to fit)
fit=Gradient_descent(normX,Y,Theta)

#conclusion
print "****************************************************"
print "the Thetas should be: ", Theta[0,0], Theta[1,0], Theta[2,0]
print "Cost function = ", CostFunc(normX,Y,Theta)
print "****************************************************"

ax.scatter(X[:,1],X[:,2], fit, c='r')
plt.savefig('linearregression_multivariable.png')
plt.show()
plt.close(fig1)

#############################################################
'''plot Cost Function'''
fig2=plt.figure(2)
plt.plot(np.array(cost_history))
plt.xlim(1,1000)
plt.xlabel('Iterations')
plt.ylabel('Cost function')
plt.savefig('Costfunction_multipvariables.png')
plt.show()

###########################################################
'''Give Prediction'''
Xtest=np.array([1.0, 1650.0, 3.0])
Xtest.shape=(1, 3)
for i in range(1,3):
    Xtest[0,i] = (Xtest[0,i]-mu[i-1])/sigma[i-1]

print "for 1650 sq-ft and 3 br hourse, the predicted price is: ", h(Xtest, Theta)

