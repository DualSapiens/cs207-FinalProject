import sys
sys.path.append("../../autodiff")
sys.path.append("../../therapy_planner/therapy_planner")
from costfunctions import *
import numpy as np
from autodiff.autodiff import Var
from autodiff.math import *
from .bfgs import BFGS

def read_maps(filename):
    """
    :param filename: The filename of the text file where user defined the therapy maps (as in the format of the demo.map)
    :return: maps, where key is "target", "max" or "min" depending on the map type and value is an numpy 2D array

    See Interface Demo.ipynb for usage
    """
    def get_map_type(line):
        if "target" in line.lower():
            return "target"
        elif "max" in line.lower():
            return "max"
        elif "min" in line.lower():
            return "min"
        else:
            raise Exception('Invalid Map Type')

    def process_map_lines(map_lines):
        n_row = len(map_lines)
        n_col = len(map_lines[0].split(','))
        therapy_map = np.zeros((n_row, n_col))
        for i, line in enumerate(map_lines):
            values = [int(x) for x in line.split(',')]
            for j, value in enumerate(values):
                therapy_map[i][j] = value
        return therapy_map

    type = None
    map_lines = []
    maps = {}
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if len(line) == 0: # empty line which is separator of maps
                therapy_map = process_map_lines(map_lines)
                maps[type] = therapy_map
                map_lines = []
            elif line[0] == "#":
                type = get_map_type(line)
            else:
                map_lines.append(line)
        if len(map_lines) > 0:
            therapy_map = process_map_lines(map_lines)
            maps[type] = therapy_map
    return maps

def optimize(maps, penalty=False, smoothness=1., tol=1e-8, maxiter=1000):
    """
    :param maps: dictionary of target, min, and max dose maps
    :param penalty: whether to include a penalty term (regularizer) to cost function penalizing source values < 0
    :param smoothness: the smoothness of the logistic function used in the penalty term
    :param tol: stopping criterion for step size in optimization
    :param maxiter: maximum number of iterations in optimization

    :return sources: the source intensities, listed first down all rows, then across all columns
    :return dose_map: the resulting dose map at the optimal source intensities, of the same shape as the input maps

    See optimize_demo.ipynb for example
    """
    # check that all maps are the same size
    if not (maps['target'].shape == maps['min'].shape and maps['target'].shape == maps['max'].shape):
        raise Exception('All maps must have the same shape.')
    else:
        m,n = maps['target'].shape
        I = np.zeros((m*n,m+n)) # matrix of intensity factors
        mu = 0.07 # attenuation coefficient in human tissue (brain, lung, blood) for 1 MeV photon energy
                  # (Ref: http://dergipark.gov.tr/download/article-file/131798)
        # Populate intensity factors
        for i in range(m):
            I[i*n:(i+1)*n,i] = np.exp(-mu*np.arange(n))
        for j in range(n):
            I[j:m*n:n,m+j] = np.exp(-mu*np.arange(m))

        S = np.array([Var() for _ in range(m+n)]) # array of source intensities
        D = np.dot(I,S) # Array of radiation doses
        Do = np.ravel(maps['target']) # Array of target doses
        
        cost = mean_squared_error(D,Do)
        if penalty:
            cost += positive_params(S,smoothness)
        cost+=minmaxpenalty(np.ravel(maps['min'])-D)
        cost+=minmaxpenalty(D-np.ravel(maps['max']))
            
        step, Niter = BFGS(cost,S,np.ones(len(S)),tol=tol,maxiter=maxiter)

        sources = [s.value for s in S] # extract fitted parameters
        dose_map = np.array([d.value for d in D]).reshape(m,n)

        return sources, dose_map


