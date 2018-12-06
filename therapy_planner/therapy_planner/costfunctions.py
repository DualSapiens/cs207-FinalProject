# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:16:01 2018

@author: gk
"""
import sys
sys.path.append("../../gradpy")
from gradpy.math import *

#Define an approximate step function with a tunable sharpness parameter
def approximate_step_function(x,sharpness=5):
    return Logistic(x,x_0=0.5,L=2,k=sharpness)

# a penalty function to encourage params >= 0
def pos_penalty(p,smoothness):
    return sum([Logistic(-pi,k=1./smoothness) for pi in p])

# mean squared error for the target parameter
def mean_squared_error(y,yo):
    return sum([(yi-yio)**2 for yi,yio in zip(y,yo)])

# Minimax penalty using the approximate step function
def minmax_penalty(vec):
    return sum(approximate_step_function(x) for x in vec)

