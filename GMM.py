# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random
from sklearn.datasets import make_blobs

epoch = 1
k = 3
N = 150
T = 50
means     = np.array([0, 5, 10]).astype(np.float64)
variances = np.array([1., 1., 1.]).astype(np.float64)
pprobs    = np.array([1/3, 1/3, 1/3]).astype(np.float32)
z         = np.random.normal(means[0], np.sqrt(variances[0]), (N, k)).astype(np.float64)
x = np.empty(0)
for i in range(50):
    n = random.gauss(0, 1)
    x = np.append(x, n)

for i in range(50):
    n = random.gauss(5, 1)
    x = np.append(x, n)
    
for i in range(50):
    n = random.gauss(10, 1)
    x = np.append(x, n)
    
x,y = make_blobs(cluster_std=[1.5, 2.5, 4],random_state=42, n_samples=150, centers=[[0], [5], [10]])

diff_means = np.empty(0).astype(np.float64)
diff_var   = np.empty(0).astype(np.float64)
    
def calculateExpectedLabels():
    global pprobs, means, variances, z, x, N, y
    for i in range(N):
        y[i] = np.argmax(z[i])
    
def updateMeans(i):
     global pprobs, means, variances, z, x, N, diff_means
     zt = np.transpose(z)
     n  = zt[i] * x[i]
     means[i] = n.sum()/zt[i].sum()
    
def updateVariances(i):
     global pprobs, means, variances, z, x, N, diff_var
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
    global pprobs, means, variances, z, diff_means, diff_var
    old_means = means.astype(np.float64)
    old_vars = variances.astype(np.float64)
    for i in range(k):
        updateProbs(i)
        updateMeans(i)
        updateVariances(i)
    diff_means = means - old_means
    diff_var = variances - old_vars

def gmm():
    global means, variances, pprobs, N, k, z
    for i in range(N):
        for j in range(k):
            z[i][j] = calculateProbs(i, j)
        z[i] = calculateRowSum(z[i])
        
def main():
    global x, means, variances, diff_means, diff_var
    for i in range(epoch):
        print("old means        : ", means)
        print("old variances    : ", variances)
        gmm()
        updateVals()
        print("updated means    : ", means)
        print("updated variances: ", variances)
        print("diff_means: ", diff_means)
        print("diff_var: ", diff_var)
        print("--------------------------------")

    #calculateExpectedLabels()
    
if __name__ == "__main__":
    main()