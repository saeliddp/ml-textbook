# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:05:06 2021

@author: saeli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
# builds a regression model with batch gradient descent
# and returns a list of vectors of estimated coefficients for each 
# predictor parameter, ending in the most recently estimated vector,
# which should be used as the model
#
# parameters:
#    X         - columns for predictor variables
#    y         - column for outcome variable
#    alpha     - learning rate ( in [0, 1] )
#    epsilon   - minimum change in cost from step i to i + 1
#                in order to continue ( in [0, 1) )
def bgd(y, X, alpha, epsilon):
    # initialize our "guess" for each coefficient theta_j to 1
    # and store these in a single column
    theta = np.ones(shape=(X.shape[1], 1))
    # to keep track of historical values of theta
    thetas = [theta]
    
    m = X.shape[0] # number of data points
    
    # calculate a column of predicted y values for each data point
    y_hat = X @ theta
    
    # calculate a 1 by 1 matrix that holds the sum of the squared
    # differences between each y_hat and y
    cost = np.transpose(y_hat - y) @ (y_hat - y)
    # initialize list of costs to contain the cost associated with
    # our initial coefficients (scaled by 1/2m in accordance with
    # the cost formula)
    costs = [cost[0][0] / (2 * m)]
    
    i = 0 # number of iterations
    delta = 1 # change in cost
    
    while (delta > epsilon):
        
        # calculate a column that holds the difference between y_hat and
        # y for each data point
        differences = X @ theta - y
        
        # update each theta_j by the partial derivative of the cost with
        # respect to theta_j, scaled by learning rate
        # Note: np.transpose(X) gives us the observed values (x_j) for
        #       a parameter j in the j'th row of a matrix
        theta = theta - (alpha / m) * ((np.transpose(X)) @ differences)
        thetas.append(theta)
        # using the updated coefficient values, append the new cost value
        cost = np.transpose(X @ theta - y) @ (X @ theta - y)
        costs.append(cost[0][0] / (2 * m))
        delta = abs(costs[i + 1] - costs[i])
        
        if (costs[i + 1] > costs[i]):
            print('Cost is increasing. Try reducing alpha.')
            break
        i += 1
        
    print('Completed in', i, 'iterations.')
    return thetas

# set up our predictor / response columns
df = pd.read_csv('regression.csv')
X = df[['x']]
y = df[['y']]

# set up our plotting environment
plt.xlabel('x')
plt.ylabel('y')
plt.title('Batch Gradient Descent')

xt = np.arange(0, df.x.max() + 5, 5)
yt = np.arange(0, df.y.max() + 10, 10)

plt.axis([xt[0], xt[-1], yt[0], yt[-1]])
plt.xticks(xt[1:])
plt.yticks(yt[1:])

# create a scatter plot of the original data
plt.scatter(df.x, df.y, facecolors='none', edgecolors='lightgray')

# define endpoints for regression lines based off of x-axis tick limits
x_prime = [xt[0], xt[-1]]

# plots a current fit using the passed in column of coefficient
# values, theta. Assumes that theta has only one element
def plot_model(theta, color='dimgrey'):
    y_hat = [xp * theta[0] for xp in x_prime]
    plt.plot(x_prime, y_hat, color=color)
    
# for simplicity, we won't add a constant term to X for this 
# gradient descent visualization
X = X.to_numpy()
y = y.to_numpy()


thetas = bgd(y=y, X=X, alpha=0.001, epsilon=10**-5)
for theta in thetas:
    plot_model(theta, color=(1,0,0,0.15))


plt.savefig('bgd_process.png', dpi=300)