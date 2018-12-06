# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:27:42 2018

@author: gk
"""
import time
def timer(func):
    def wrapper():
        ts = time.time()
        func()
        te = time.time()
        print((ts-te))
    return wrapper