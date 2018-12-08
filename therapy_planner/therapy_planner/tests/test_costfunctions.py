# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:18:10 2018

@author: gk
"""

import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
from gradpy.math import *
from costfunctions import *

class TestCostfunctions:
    def testmeansquarederror(self):
        assert np.isclose(mean_squared_error([5,6,5],[6,1,3]),30)
        
    def testminmax_penalty(self):
        assert np.isclose(minmax_penalty([0.1]),1.24491)