import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cost(X,y,theta):
    m = len(X)
    h_x  = X@theta
    J = np.sum((h_x - y)**2) / (2 * m)
    return J


def gradientDescentFor(X, y, theta, alpha,num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1));
    for i in range(num_iters):
        error= []
        for x,targtt in zip(X,y):
            error.append(np.sum(x*theta)- targtt)
        cal_err=np.transpose(X) @ error
        for i in range(len(theta)):
            theta[i] = theta[i] - ((alpha/m)*(np.sum(cal_err[i],axis=0)))
        J_history[i] = cost(X, y, theta)
        print(J_history[i])
    return theta,J_history

def gradientDescent(X, y, theta, alpha,num_iters):
    m=len(y)
    J_history = np.zeros((num_iters, 1));
    for i in range(num_iters):
        error = (X @ theta) - y
        theta = theta - ((alpha/m) * (np.transpose(X)@error))
        J_history[i] = cost(X, y, theta)
    return theta, J_history
  
  
def plotData(X,y):
    plt.scatter(X,y,marker='x')
    plt.show()

def plotDataPredictions(X,y,theta):
    m = X.shape[0]
    Xp = np.hstack((np.ones((m,1)),X))
    predictions = Xp @ theta
    plt.scatter(X,y,marker='x')
    plt.plot(X,predictions)
    plt.show()



df = pd.read_csv('ex1data1.txt')
y = df.y.values
X1 = df.drop(columns = ['y'])
X1.insert(0,'X0',1) #with X0
X =X1
Xplot = df.drop(columns = ['y'])
m = X.shape[1]
initial_theta = np.zeros(m)
alpha = 0.01
num_iters = 1000
result_theta, J_his = gradientDescent(X,y,initial_theta,alpha,num_iters)
cost_ = cost(X,y,initial_theta)

# plotData(Xplot,y)
plotDataPredictions(Xplot,y,result_theta)