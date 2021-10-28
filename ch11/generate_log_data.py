# -*- coding: utf-8 -*-
"""
@author: saeli
"""
import csv
import random

# customer group: {introduction #: probability of responding to introduction, ...}
cgroup_response_probs = {
    0: {0: .6, 1: .2, 2: .4},
    1: {0: .8, 1: .1, 2: .2},
    2: {0: .5, 1: .5, 2: .4},
    3: {0: .4, 1: .3, 2: .6},
    4: {0: .1, 1: .4, 2: .7},
    5: {0: .2, 1: .3, 2: .5}
}

def get_customer_group(age, logins):
    if age <= 25 and logins <= 5:
        return 0
    elif age <= 25:
        return 1
    elif age <= 50 and logins <= 5:
        return 2
    elif age <= 50:
        return 3
    elif logins <= 5:
        return 4
    else:
        return 5

with open('log_data.csv', 'w', newline='') as fw:
    writer = csv.writer(fw)
    writer.writerow(['age','logins','introduction','responded'])
    for _ in range(2000):
        age = random.randint(17, 85)
        logins = random.randint(1, 20)
        intro_r = random.random()
        if intro_r < 0.5:
            intro = 0
        elif intro_r < 0.75:
            intro = 1
        else:
            intro = 2
            
        cgroup = get_customer_group(age, logins)
        if random.random() < cgroup_response_probs[cgroup][intro]:
            responded = 1
        else:
            responded = 0
        
        writer.writerow([age, logins, intro, responded])
            
        