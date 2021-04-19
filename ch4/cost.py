# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:54:43 2021

@author: saeli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# performs batch gradient descent, as in the previous example,
# but returns a list of costs rather than the thetas associated with
# those costs
def batch_gradient_descent(y, X, alpha, epsilon):
    theta = np.ones(shape=(X.shape[1], 1))
    m = X.shape[0]
    
    cost = np.transpose(X @ theta - y) @ (X @ theta - y)
    costs = [cost[0][0] / (2 * m)]
    
    i = 0
    delta = 1
    
    while (delta > epsilon):
        theta = theta - (alpha / m) * ((np.transpose(X)) @ (X @ theta - y))
                
        cost = np.transpose(X @ theta - y) @ (X @ theta - y)
        costs.append(cost[0][0] / (2 * m))
        delta = abs(costs[i + 1] - costs[i])
        
        if (costs[i + 1] > costs[i]):
            print('Cost is increasing. Try reducing alpha.')
            break
        i += 1
        
    print('Completed in', i, 'iterations.')
    return costs

# set up our predictor / response columns
df = pd.read_csv('regression.csv')
X = df[['x']].to_numpy()
y = df[['y']].to_numpy()

# set up our plotting environment
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('Cost Function')
costs = batch_gradient_descent(y=y, X=X, alpha=0.001, epsilon=10**-5)
plt.plot([i for i in range(len(costs))], costs)

plt.savefig('cost.png', dpi=300)