import sys
sys.path.append("../../gradpy")
import numpy as np
from gradpy.autodiff import Var
from gradpy.math import *
from .bfgs import BFGS
from .costfunctions import *

class PlannerInterface:
    def __init__(self, filename):
        self.datafile = filename
        self.read_maps()
        if not (self._maps['target'].shape == self._maps['min'].shape and self._maps['target'].shape == self._maps['max'].shape):
            raise Exception('All maps must have the same shape.')
        else:
            self.shape = self._maps['target'].shape
        self.opt = False  # flag if plan has been optimized

    def read_maps(self):
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
        with open(self.datafile, 'r') as f:
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
        self._maps = maps

    def get_maps(self):
        return self._maps

    def optimize(self, exposure_time, smoothness=1., tol=1e-8, maxiter=1000, bounds=False):
        """
        :param maps: dictionary of target, min, and max dose maps
        :param exposure_time: the duration over which to deliver the radiation dose
        :param smoothness: the smoothness of the logistic function used in the penalty term
        :param tol: stopping criterion for step size in optimization
        :param maxiter: maximum number of iterations in optimization

        :return profile_values: the accumulated intensity profiles at each beam location,
                                listed first down all rows, then across all columns
        :return dose_map: the accumulated dose map at the optimal intensity profiles, of the same shape as the input maps

        See optimize_demo.ipynb for example
        """
        m,n = self.shape
        self.exp_time = exposure_time

        # Step 1: Optimize intensity profiles for each beam.
        profiles, dose = self.optimize_intensity_profiles(smoothness, tol, maxiter, bounds)
        
        # Step 2: Compute the beam intensity and sequence of apertures from the optimized profiles.
        self.compute_aperture_sequences(profiles)
        
        # Collect values for output.
        profile_values = [p.value for p in profiles]
        dose_map = np.array([d.value for d in dose]).reshape(m,n)
        self.profiles = profile_values
        self.dose_map = dose_map
        self.opt = True # set optimization flag
        return self.beam, profile_values, dose_map, self.horiz_collimator, self.vert_collimator

    def optimize_intensity_profiles(self, smoothness, tol, maxiter, bounds):
        m,n = self.shape
        attenuation = np.zeros((m*n,m+n)) # matrix of attenuation factors
        mu = 0.07 # attenuation coefficient in human tissue (brain, lung, blood) for 1 MeV photon energy
                  # (Ref: http://dergipark.gov.tr/download/article-file/131798)
        for i in range(m):
            attenuation[i*n:(i+1)*n,i] = np.exp(-mu*np.arange(n))
        for j in range(n):
            attenuation[j:m*n:n,m+j] = np.exp(-mu*np.arange(m))

        profiles = np.array([Var() for _ in range(m+n)]) # intensity profiles in vertical and horizontal directions
        dose = np.dot(attenuation,profiles) # accumulated dose in each grid cell
        cost = mean_squared_error(dose,np.ravel(self._maps['target'])) \
             + pos_penalty(profiles,smoothness)
        if bounds:
            cost+=minmax_penalty(np.ravel(self._maps['min'])-dose)
            cost+=minmax_penalty(dose-np.ravel(self._maps['max']))
           
        step, Niter = BFGS(cost,profiles,np.ones(len(profiles)),tol=tol,maxiter=maxiter)
        return profiles, dose

    def compute_aperture_sequences(self, profiles):
        m,n = self.shape
        profile_values = [p.value for p in profiles] # extract fitted intensity values
        beam = np.max(profile_values)/self.exp_time
        self.beam = beam
        horiz_profile = np.array([0] + [(p//beam+np.round(p%beam)/beam).astype(np.int) for p in profile_values[:m]] + [0],dtype=np.int)
        vert_profile = np.array([0] + [(p//beam+np.round(p%beam)/beam).astype(np.int) for p in profile_values[m:]] + [0],dtype=np.int)
        self.horiz_collimator = {"left": [],"right": []}
        self.vert_collimator = {"left": [],"right": []}
        keys = ["left","right"]
        x = 0
        for a,b in zip(horiz_profile[:-1],horiz_profile[1:]):
            self.horiz_collimator[keys[(a>b).astype(np.int)]] += [x for _ in range(min([a,b]),max([a,b]))]
            x += 1
        x = 0
        for a,b in zip(vert_profile[:-1],vert_profile[1:]):
            self.vert_collimator[keys[(a>b).astype(np.int)]] += [x for _ in range(min([a,b]),max([a,b]))]
            x += 1
        profile_values = beam*np.hstack([horiz_profile[1:-1],vert_profile[1:-1]]) # set new profile values incorporating beam resolution
        for p,val in zip(profiles,profile_values):
            p.set_value(val)

    def print_summary(self):
        if not self.opt:
            raise Exception("No summary available; plan has not been optimized.")
        else:
            total_dose = np.sum(self.dose_map)
            avg_dose = np.mean(self.dose_map)
            contents = "Beam intensity: "+"%.2f mW/cm^2"%self.beam+"\n"+ \
                       "Horizontal beam exposure time: "+"%d sec."%len(self.horiz_collimator["left"])+"\n"+ \
                       "Vertical beam exposure time: "+"%d sec."%len(self.vert_collimator["right"])+"\n" \
                       "Total accumulated dose: "+"%.2f Gy"%total_dose+"\n" \
                       "Average dose per unit area: "+"%.2f Gy/cm^2"%avg_dose+"\n"
            print(contents)



