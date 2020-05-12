# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random

epoch = 1
k = 3
N = 150
T = 50
means     = np.array([1, 4, 9])
variances = np.array([1, 1, 1])
pprobs    = np.array([1/3, 1/3, 1/3])
z         = np.random.normal(means[0], np.sqrt(variances[0]), (N, k))
x1        = np.random.normal(means[0], np.sqrt(variances[0]), 50)
x2        = np.random.normal(means[1], np.sqrt(variances[1]), 50)
x3        = np.random.normal(means[2], np.sqrt(variances[2]), 50)
x         = np.array(list(x1) + list(x2) + list(x3))
y         = np.zeros((N, 1))

def calculateExpectedLabels():
    global pprobs, means, variances, z, x, N, y
    for i in range(N):
        y[i] = np.argmax(z[i])
    
def updateMeans(i):
     global pprobs, means, variances, z, x, N
     zt = np.transpose(z)
     n  = zt[i] * x[i]
     means[i] = n.sum()/zt[i].sum()
    
def updateVariances(i):
     global pprobs, means, variances, z, x, N
     zt = np.transpose(z)
     n  = zt[i] *  np.abs(x[i]-means[i])
     variances[i] = n.sum()/zt[i].sum()

def updateProbs(i):
     global pprobs, means, variances, z, x, N
     zt = np.transpose(z)
     pprobs[i] = zt[i].sum()/N
 
def calculateRowSum(x):
    for i in range(k):
        x[i] /= np.sum(x)
    return x

def calculateProbs(i, j):
    global pprobs, means, variances, z, x
    t  = 2 * np.pi * variances[j]
    n1 = pprobs[j] * (1/np.sqrt(t))
    n2 = np.exp(-(np.square(x[i]-means[j]))/(2*variances[j]))
    n = n1 * n2
    return n
   
def updateVals():
    global pprobs, means, variances, z
    for i in range(k):
        updateProbs(i)
        updateMeans(i)
        updateVariances(i)
        

def gmm():
    global means, variances, pprobs, N, k, z
    for i in range(N):
        for j in range(k):
            z[i][j] = calculateProbs(i, j)
        z[i] = calculateRowSum(z[i])
    print("--------------------------------")
        
def main():
    global x, means, variances
    for i in range(epoch):
        print("old means        : ", means)
        print("old variances    : ", variances)
        gmm()
        updateVals()
        print("updated means    : ", means)
        print("updated variances: ", variances)

    #calculateExpectedLabels()
    
if __name__ == "__main__":
    main()