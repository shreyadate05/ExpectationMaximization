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
means     = np.random.uniform(low=0.5, high=13.3, size=(k, 1))
variances = np.random.uniform(low=0.5, high=13.3, size=(k, 1))
pprobs    = np.full((k, 1), 1/k)
x         = np.random.normal(0.0, pow(1, -0.5),(N, 1))
z         = np.random.normal(0.0, pow(1, -0.5),(N, k))
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
     n  = zt[i] *  np.linalg.norm(x[i]-means[i])
     variances[i] = n.sum()/zt[i].sum()

def updateProbs(i):
     global pprobs, means, variances, z, x, N
     zt = np.transpose(z)
     pprobs[i] = zt[i].sum()/N
 
def calculateRowSum(z):
    for row in range(z.shape[0]):
        z[row] /= np.sum(z[row])
    return z

def calculateProbs(i, j):
    global pprobs, means, variances, z, x
    t  = 2 * np.pi * variances[j]
    n1 = pprobs[j] * (1/np.sqrt(t))
    n2 = np.exp((-1/(2*variances[j])) * np.linalg.norm(np.square(x[i]-means[j])))
    return n1*n2
   
def updateVals():
    global pprobs, means, variances
    for i in range(k):
        updateMeans(i)
        updateVariances(i)
        updateProbs(i)

def gmm():
    global means, variances, pprobs, N, k, z
    for i in range(N):
        for j in range(k):
            z[i][j] = calculateProbs(i, j)
        z[i] = calculateRowSum(z[i])
        
def main():
    for i in range(epoch):
        gmm()
        updateVals()
    calculateExpectedLabels()
    
if __name__ == "__main__":
    main()