# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:27:42 2018

@author: gk
"""
import time
def timer(func):
    def wrapper(*args, **kw):
        ts = time.time()
        func(*args, **kw)
        te = time.time()
        print("Time Elapsed: %.4f sec." % (te-ts))
    return wrapper