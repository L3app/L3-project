#!/usr/bin/env python
from scipy.optimize import differential_evolution
import subprocess
import csv
import rospy
import math
import sys
import time

def F(x):
    neu_dict = {'PID': []}
    neu_dict['PID'].append(x[0])
    neu_dict['PID'].append(x[1])
    neu_dict['PID'].append(x[2])
	
    with open("test.csv", "wb") as f:
     writer = csv.writer(f)
     writer.writerow(neu_dict.keys())
     writer.writerows(zip(*neu_dict.values()))
    
    subprocess.call(['./bash.sh'])
    
    with open("test1.csv") as f:
     lis=[line.split(',') for line in f] 
    
    lis1 = [float(x[0].rstrip()) for x in lis[1:len(lis) - 1]]
    lis2 = [float(x[1].rstrip()) for x in lis[1:len(lis) - 1]]
    
    sum1 = lis1[0] * lis2[0] * ((lis1[1] - lis1[0]) * (10 ^ (-9)))
    for i in range(len(lis1) - 1):
        sum1 = sum1 + (lis1[i + 1] * lis2[i + 1] * ((lis1[i + 1] - lis1[i])) * (10 ^ (-9)))
    error = sum1
    print error

    return error


def main():
    x0 = [-114.05, -0.9995, -2891.38]
    bounds = [(-3000,3000), (-3000,3000), (-3000,3000)]
    res = differential_evolution(F, bounds, maxiter=0, popsize=1)
    print res[0]

    





if __name__ == '__main__':
    main()
