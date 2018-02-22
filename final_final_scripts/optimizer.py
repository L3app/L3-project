#!/usr/bin/env python
from scipy.optimize import differential_evolution
from subprocess import *
import subprocess
import csv
import rospy
import math
import sys
import time

def F(x):
    global iteration
    neu_dict = {'PID': []}
    neu_dict['PID'].append(x[0])
    neu_dict['PID'].append(x[1])
    neu_dict['PID'].append(x[2])
    print neu_dict
    print "Iteration number ", iteration

    #print x
	
    with open("test.csv", "wb") as f:
    	writer = csv.writer(f)
    	writer.writerow(neu_dict.keys())
    	writer.writerows(zip(*neu_dict.values()))
    

    subprocess.call(['./bash.sh'])
    #subproc = Popen(['./bash.sh'], stdout=PIPE, stderr=PIPE)
    #(out, err) = subproc.communicate()
    

    with open("test1.csv") as f:
    	lis = [line.split(',') for line in f] 
    
    	inst_e = [float(x[0].rstrip()) for x in lis[1:len(lis) - 1]]
    	inst_t = [float(x[1].rstrip()) for x in lis[1:len(lis) - 1]]

    	sum1 = inst_e[0] * inst_t[0] * ((inst_t[1] - inst_t[0]))
	#print("First sum gives: ", sum1)
    for i in range(len(inst_e) - 1):
	sum1 = sum1 + (inst_e[i + 1] * inst_t[i + 1] * ((inst_t[i + 1] - inst_t[i])))
    #print("Second sum gives: ",sum1)
    error = sum1
    
    with open("test.csv") as h:
    	lis = [line.split(',') for line in h]
	#print lis 
    	x = [float(line[0].rstrip()) for line in lis[1:len(lis)]]
    	#print x
    rec_dict = {'P': [],'I': [], 'D': [], 'error': [],'iteration': []}
    rec_dict['P'].append(x[0])
    rec_dict['I'].append(x[1])
    rec_dict['D'].append(x[2])
    rec_dict['error'].append(error)
    rec_dict['iteration'].append(iteration)

    #print rec_dict
    
    

    with open("iter_res.csv", "a") as g:
    	writer = csv.writer(g)
    	writer.writerows(zip(*rec_dict.values()))

    rec_dict.clear()

    print("The error for these PID-values is ", error)

    iteration = iteration + 1

    return error


def main():
    #global rec_dict
    global iteration
    iteration = 1
    rec_dict = {'P': [],'I': [], 'D': [], 'error': [],'iteration': []}
    with open("iter_res.csv", "wb") as g:
    	writer = csv.writer(g)
    	writer.writerow(rec_dict.keys())

    x0 = [-114.05, -0.9995, -2891.38]
    #bounds = [(-3000,3000), (-3000,3000), (-3000,3000)]
    #bounds = [(0,0), (0.01,0.01), (1.1,1.1)] this is good
    bounds = [(0,0.2), (0,0.3), (0.5,1.5)]
    res = differential_evolution(F, bounds, maxiter=100, popsize=10,disp=True)
    print("The outputs from optimizer main are: ")
    print("Solution x= ", res.x)
    print("Success bool: ", res.success)
    #print("Status: ", res.status)
    print("Message: ", res.message)
    print("Function minimum found, error = ", res.fun)
    print("Number of evaluations: ", res.nfev)
    print("Number of iterations performed: ", res.nit)
    
    





if __name__ == '__main__':
    main()
