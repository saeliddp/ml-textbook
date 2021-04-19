# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:38:36 2021

@author: saeli
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('regression.csv')

X = df.x
y = df.y

X = sm.add_constant(X)

lr_model = sm.OLS(y, X).fit()

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression (OLS)')

# define lists of x and y axis ticks
xt = np.arange(0, df.x.max() + 5, 5)
yt = np.arange(0, df.y.max() + 10, 10)

# set ticks and axis limits for the plot
plt.axis([xt[0], xt[-1], yt[0], yt[-1]])
plt.xticks(xt[1:])
plt.yticks(yt[1:])

# create a scatter plot of the original data
plt.scatter(df.x, df.y, facecolors='none', edgecolors='lightgray')

# define endpoints for regression line based off of x-axis tick limits
x_prime = [xt[0], xt[-1]]

# add constant term and plot predicted y-values based off of the model
x_prime = sm.add_constant(x_prime)
y_hat = lr_model.predict(x_prime)
plt.plot(x_prime[:, 1], y_hat)
plt.savefig('regression.png', dpi=300)


