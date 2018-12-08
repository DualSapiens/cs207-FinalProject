# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:18:10 2018

@author: gk
"""

import sys
sys.path.append("../../../gradpy")

from ..costfunctions import *
from gradpy.math import *
import numpy as np

class TestCostfunctions:
    def testmeansquarederror(self):
        assert np.isclose(mean_squared_error([5,6,5],[6,1,3]),30)
        
    def testminmax_penalty(self):
        assert np.isclose(minmax_penalty([0.1], smoothness=0.2).value,0.23840584404423515)