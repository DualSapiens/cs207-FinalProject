# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:16:01 2018

@author: gk
"""
import sys
sys.path.append("../../autodiff")
sys.path.append("../../therapy_planner/therapy_planner")

from autodiff.autodiff import Var
from autodiff.math import *

# define some cost functions (can go into their own module)
def mean_squared_error(y,yo):
    return sum([(yi-yio)**2 for yi,yio in zip(y,yo)])

# a penalty function to encourage params >= 0
def positive_params(params,smoothness):
    return sum([Logistic(-p,k=1./smoothness) for p in params])