# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:18:10 2018

@author: gk
"""

from ..costfunctions import *
from gradpy.math import *
import numpy as np

class TestCostfunctions:
    def test_mean_squared_error(self):
        assert np.isclose(mean_squared_error([5,6,5],[6,1,3]), 30)
        
    def test_minmax_penalty(self):
        assert np.isclose(minmax_penalty([0.1], [0], smoothness=0.2).value, 0.23840584404423515)

    def test_nan(self):
    	assert np.isclose(mean_squared_error([5,6,18,5,12],[6,1,np.nan,3,np.nan]), 30)
    	assert np.isclose(minmax_penalty([0.1, 10.], [0, np.nan], smoothness=0.2).value, 0.23840584404423515)