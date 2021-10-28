# -*- coding: utf-8 -*-
"""
@author: saeli
"""
import pandas as pd
import numpy as np

# given the customer age and number of logins, return the introduction
# choice that the new system would have made
def new_system_choice(age, logins):
    if age <= 25:
        return 0
    elif age <= 50 and logins > 5:
        return 1
    else:
        return 2

# calculates the IPS contribution of a single row based on what the new
# system would have done
def row_inverse_score(row):
    new_choice = new_system_choice(row['age'], row['logins'])
    if new_choice == row['introduction']:
        return 1 / row['propensity']
    else:
        return 0

logs = pd.read_csv('log_data.csv')
# calculate response rate for current system
current_resp_rate = np.mean(logs['responded'])
print(f'current response rate: {current_resp_rate}')

# add propensity column
intro_props = {0: 0.5, 1: 0.25, 2: 0.25}
logs['propensity'] = logs.apply(lambda row: intro_props[row['introduction']], axis=1)

# calculate the IPS for your new system
row_contribs = logs.apply(lambda row: row_inverse_score(row), axis=1)
ips_new_system = np.mean(row_contribs)
print(f'estimated new response rate (IPS): {ips_new_system}')
