# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:57:34 2021

@author: saeli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# builds a regression model with online stochastic gradient descent
# and returns a list of vectors of estimated coefficients for each 
# predictor parameter, ending in the most recently estimated vector,
# which should be used as the model
#
# parameters:
#    X         - columns for predictor variables
#    y         - column for outcome variable
#    alpha     - learning rate ( in [0, 1] )
#    n_itr     - used to fix the number of iterations to run
def sgd(y, X, alpha, n_itr=50):
    # initialize our "guess" for each coefficient theta_j to 1
    # and store these in a single column
    theta = np.ones(shape=(X.shape[1], 1))
    # to keep track of historical values of theta
    thetas = [theta]
    
    m = X.shape[0] # number of data points
    
    # will hold rows that have not yet been used by this epoch of sgd
    epoch = []
    # initialize list of costs 
    costs = []
    
    for i in range(n_itr):
        if len(epoch) == 0:
            epoch = [row for row in range(m)]
            
        curr_row = epoch.pop(np.random.randint(len(epoch)))
        # calculate a predicted y value for the randomly chosen row
        y_hat = (X[curr_row] @ theta)[0]
        difference = y_hat - y[curr_row]
        
        # in sgd, cost is simply the scaled square error for our currently chosen row
        cost = difference**2 / (2 * m)
        costs.append(cost)
        # update each theta_j by the partial derivative of the cost with
        # respect to theta_j, scaled by learning rate
        updates = [(alpha / m) * difference * X[curr_row]]
        theta = theta - updates
        thetas.append(theta)
        
        
    print('Completed in', n_itr, 'iterations.')
    return thetas

# set up our predictor / response columns
df = pd.read_csv('regression.csv')
X = df[['x']]
y = df[['y']]

# set up our plotting environment
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stochastic Gradient Descent')

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

# notice that with a small number of iterations and a relatively large alpha,
# we get some diversity in the final fit due to the stochastic nature of this
# algorithm
for i in range(4):
    plot_model(sgd(y=y, X=X, alpha=0.05, n_itr=10)[-1], color=((i+1)*0.2,0,0))
    
plt.savefig('sgd.png', dpi=300)