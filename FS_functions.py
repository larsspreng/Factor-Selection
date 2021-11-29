# -*- coding: utf-8 -*-
"""
Written by: Lars Spreng

File contains functions to estimate optimal number of factors for large datasets
"""
import pandas as pd
import numpy as np
import numpy.linalg as la

# Input variables
# mX:   Dataset (T x N)
# kmax: Maximum number of factors allowed (1 <= kmax <= N)

# Output
# r0:   Optimal number of factors according to criterion

def AH_crit_ER(mX,kmax):
    # Computes optimal number of factors according to Eigenvalue Ratio
    # criterion in Ahn, Horenstein (2013, ECTA)
    T, N = mX.shape
    Z = np.divide(mX.T.dot(mX.values),T*N)
    eigvals = la.eigvals(Z)
    eigvals = eigvals[0:kmax]
    ER = np.divide(eigvals[0:kmax-1],eigvals[1:kmax])
    r0 = np.argmax(ER) + 1
    return r0

def AH_crit_GR(mX,kmax):
    # Computes optimal number of factors according to Growth Ratio
    # criterion in Ahn, Horenstein (2013, ECTA)
    T, N = mX.shape
    Z = np.divide(mX.T.dot(mX.values),T*N)
    eigvals = la.eigvals(Z)
    V = np.zeros(shape=(kmax,1))
    for i in range(2,kmax+2):
        V[i-2] = np.sum(eigvals[i-1:])
    mu_star = eigvals[0:kmax]/V.T
    GR = np.log(1+mu_star.T[0:-1])/np.log(1+mu_star.T[1:])
    r0 = np.argmax(GR) + 1
    return r0
