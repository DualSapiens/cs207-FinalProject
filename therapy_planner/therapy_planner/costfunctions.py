# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:16:01 2018

@author: gk
"""

from gradpy.math import *
import numpy as np


# Define an approximate step function with a tunable sharpness parameter
def approximate_step_function(x, smoothness):
    return Logistic(x, x_0=0.5, L=2, k=1./smoothness)  # Logistic function f(X) = L/(1+exp(-k(X-x_0)))


# a penalty function to encourage params >= 0
def pos_penalty(p, smoothness):
    return sum([Logistic(-pi, k=1./smoothness) for pi in p])


# mean squared error for the target parameter
def mean_squared_error(y, yo):
    tot = 0
    for yi, yio in zip(y, yo):
        if ~np.isnan(yio):
            tot += (yi-yio)**2
    return tot

# Minimax penalty using the approximate step function
def minmax_penalty(x, xo, smoothness=1):
    tot = 0
    for xi, xio in zip(x, xo):
        if ~np.isnan(xio):
            tot += approximate_step_function(xi-xio, smoothness)
    return tot

