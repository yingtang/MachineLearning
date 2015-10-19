import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

#define our hypothesis
def h(X, Theta):
    return np.dot(X, Theta)

def CostFunc(X,Y,Theta):
    m = Y.size
    diff = square_error = (h(X, Theta)-Y)
    J = 0.5*np.dot(diff.T, diff)/float(m)
    #square_error = (h(X, Theta)-Y)**2
    #J = 0.5*sum(square_error)/float(m)
    return J

def Gradient_descent(X,Y,Theta):
    m = Y.size
    #OldTheta=Theta
    converge=100
   
    cost= CostFunc(X, Y, Theta)

    while converge > 1E-10:
        oldCost=cost

        #print "check cost function", CostFunc(X,Y, Theta)
  
        Theta += alpha*np.dot(X.T, (Y-h(X, Theta)))/float(m)
        cost = CostFunc(X,Y,Theta)
        converge = abs(cost - oldCost)
    
    return np.array(h(X, Theta))

#visualize your data
dat=np.loadtxt('ex1data1.txt',delimiter=',')
fig1=plt.figure(1)
plt.plot(dat[:,0], dat[:,1], 'o')
plt.ylabel('Profit of a food truck')
plt.xlabel('Population in the city')


m=len(dat[:,0]) #length of the input data
fit=np.ones(m, float) 

Theta=np.zeros((2,1), float) #theta
X=np.ones((m,2),float) #features X, m x 2
Y=np.ones(shape=(m,1))  #y, m x 1

X[:,1]=dat[:,0] #x is a mX2 array
Y[:,0]=dat[:,1] #Y is a m x 1 array

#learning rate
alpha=0.01

fit=Gradient_descent(X,Y,Theta)

print "Thetas are: ", Theta[0,0], Theta[1,0]

print "Cost function = ", CostFunc(X,Y,Theta)

plt.plot(np.array(X[:,1]), fit, '+')
plt.savefig('linearregression_1variable.png')
plt.show()
plt.close(fig1)

'''Coutour Plot'''
fig2=plt.figure(2)
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta = np.zeros(shape=(2,1))
        theta[0][0]=theta0_vals[i]
        theta[1][0]=theta1_vals[j]
        J_vals[j,i]=CostFunc(X, Y, theta)

plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.scatter(Theta[0][0], Theta[1][0])
plt.savefig('cost_function_contour.png')

plt.show()
