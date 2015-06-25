'''
    Machine Learning on Coursera (Andrew Ng)
    Exercise 2.2: logistic regression with regularization 

    use scipy optimization function package scipy.optimize, package fmin_bfgs to minimize 
    the function and get the opmitized papameter (theta) values
    
    data set needed: 'ex2data2.txt'
    
    @yingtang 

note: feature normalization is not necessary here

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys


def feature_norm(x):
    '''feature normalization'''
    global mu  #mean values of each feature
    global sigma  #standard deviation of each feature
    m,n = x.shape
    xnorm = np.ones((m,n))
    
    mu=[]
    sigma=[]
    
    '''note that the first column of X are all ones'''
    for i in range(1, n):
        mu.append(np.mean(x[:,i]))
        sigma.append(np.std(x[:,i]))
        xnorm[:,i]=(x[:,i]-mu[i-1])/sigma[i-1]

    return xnorm

def unnormalize_theta(theta):
    '''
    we obtained theata with the normailzed features

    now we keep the feature normalized and get back the unormalized values of theta
    '''
    n=len(mu)
    #unnormalize theta_0
    for i in range(n):
        theta[0] -= theta[i+1]*mu[i]/sigma[i]

    #unnormalize other thetas
    for i in range(n):
        theta[i+1] = theta[i+1]/sigma[i]
        
    return theta


def boundary_lines(theta):
    '''
    plot the boudary lines of the classification
    boundary is determined when sigmoid function is of value 0.5

    for the data with two features
    '''
    num=50 #number of points

    u = np.linspace(-1.0, 2.0, num)
    v = np.linspace(-1.0, 2.0, num)

    x=np.ones((num, 3))

    xmapped = map_feature(x) #num x 28
    #theta 28 x 1

    z = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            x = np.array([1.0, u[i], v[j]])
            x.shape=(1, 3)
            xmapped = map_feature(x)  #1 x 28
            z[i, j] = np.dot(xmapped, theta)
 
    z = z.T
    plt.contour(u, v, z)
    plt.title('lambda = %f' %lambdda)
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.legend()
    plt.xlim([-1, 1.5])
    plt.ylim([-0.8, 1.2])
    plt.show()
   
        
def sigmoid(x, theta):
    """compute the sigmoid function"""
    z = np.dot(x, theta) #dimesion = m x 1 

    sgmd =  1.0/(1.0+np.exp(-z)) 
    
    #return value is m x 1
    return sgmd

def cost_function(theta, x, y):
    '''cost function of the logistic regression'''
    #when h(x) is greater than 0.5
    m,n = x.shape
    
    J_pos = -np.dot(y.T, np.log(sigmoid(x, theta)))
    J_neg = -np.dot((1.0-y).T, np.log(1.0 - sigmoid(x, theta)))

    J = (J_pos + J_neg)*(1.0/m)

    #add the regularization term
    #J += np.dot(theta[1:].T, theta[1:])*lambdda/2.0/m
    J += sum(theta[1:]**2)*lambdda/2.0/m

    return J
  
def gradient(theta,x,y):
    """compute the gradient of the cost function"""
    m, n = x.shape
    #create the gradient array
    grad = np.zeros((n,1))

    h = sigmoid(x, theta)
    h.shape=(m, 1)
    
    delta = h - y  

    grad[0] = 1.0/m*np.dot(x[:,0].T, delta)

    for i in range(1,n):
        grad[i] = (1.0/m)*np.dot(x[:,i].T, delta) + lambdda/m*theta[i]

    return grad.flatten()

                  
def visualization(x, y):
    "visualize the original data"
    admitted = np.where(y==1)
    rejected = np.where(y==0)

    plt.scatter(x[admitted, 1], x[admitted, 2], marker='+', c='k', s=30, label='accepted')
    plt.scatter(x[rejected, 1], x[rejected, 2], marker='o', c='y', s=30, label='rejected')
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend()
#    plt.show()

def map_feature(x):
    '''
    create more features from each data point. 
    here we map the features into all polynomial terms of x1 and x2 
    up to the sixth power

    note: we have two features, from zeroth order to the sixth order, there are 
    28 terms in total

    it might be overfitting, therefore we need the regularization term to 
    help combat the overfitting problems. 
    '''

    m,n = x.shape #number of the samples, and (number of original features + 1)

    degree = 6

    #create an array for the mapped features 
    mapped = np.ones((m, 1))

    #first feature
    x1 = x[:,1]

    #second feature
    x2 = x[:,2]

    for i in range(1, degree+1):
        for j in range(i+1):
            newcol = (x1**j)*x2**(i-j)
            newcol.shape = (m, 1)
            mapped = np.append(mapped, newcol, 1)

    #returm mapped feature, which is a m x 28 array in this case
    return mapped
    

def prediction(x, theta):

    print "Predict the sample with test score: \n", x
   
    x_mapped = map_feature(x)
    n = theta.size
    theta.shape = (n, 1)
    p = sigmoid(x_mapped, theta)

    print "The acceptance probability is: ", p
    if p>=0.5:
        y=1
    else:
        y=0

    return y
    
    

##############################################
#            Main Program                    #
##############################################

dat = np.loadtxt("ex2data2.txt", delimiter=',')

#get the number of rows m and number of columns n
m,n = dat.shape

#get x data
xdata = np.ones((m,n)) # m x n array
xdata[:,1:] = dat[:,0:2] # 
#get y data
ydata = dat[:,2]  #m x 1
ydata.shape=(m, 1)
 
#regularization parameter
#use "double d" to avoid the conflicted name of the "lambda" function
global lambdda 
lambdda = float(raw_input("input lambda value:"))

#check what data looks like call "visualization"
visualization(xdata, ydata)

#mapped features to certain polynomial degree, here degree = 6
xmapped = map_feature(xdata)


#normalize the mappedfeatures
#xmapped = feature_norm(xmapped)

#update n to the number of new features
n = xmapped[0,:].size 
init_theta = np.zeros((n, 1)) # n x 1

#check if the cost function is correct, should be 0.693
print "initial cost function value:", cost_function(init_theta, xmapped, ydata)

#minimize the theta sets using scipy fmin_bfgs algorithm 
theta_min, j_min = opt.fmin_bfgs(cost_function, init_theta, fprime=gradient, args=(xmapped, ydata), full_output=True, disp=False)[0:2] 

print "final theta value:", theta_min
print "the minimum cost function value:", j_min

#unnormalize theta values, not scalable in this case
#un_theta = unnormalize_theta(theta_min)
un_theta = theta_min 

#plot the boundary lines
boundary_lines(un_theta)

#Test your program, should report 0.776
xtest=[1.0, -0.25, 1.5]
xtest=np.array(xtest)
xtest.shape =  (1,3)
result=prediction(xtest, un_theta)

print("(1: admitted, 0: rejected) The chip will be: %01d"  %result)


