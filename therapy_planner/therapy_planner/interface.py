import sys
sys.path.append("../../gradpy")
import numpy as np
from gradpy.autodiff import Var
from gradpy.math import *
from .bfgs import BFGS
from .costfunctions import *

from enum import Enum

class BeamDirection(Enum):
    Horizontal = 1 # beam propagating along x
    Vertical = 2 # beam propagating along y

class Beam:
    def __init__(self, direction):
        if direction not in [BeamDirection.Horizontal,BeamDirection.Vertical]:
            raise Exception("Invalid beam direction.")
        else:
            self.direction = direction
            self.name = direction.name+" beam"

    def solve(self, beamlets, exposure_time):
        """
        Compute beam intensity and collimator aperture sequence
        based on optimized beamlets and given exposure time.
        """
        self.intensity = np.max(beamlets)/exposure_time
        profile = np.array([0] + [(b//self.intensity+np.round(b%self.intensity)/self.intensity).astype(np.int) \
                  for b in beamlets] + [0],dtype=np.int)
        keys = ["left","right"]
        self.collimator = {"left": [],"right": []}
        x = 0
        for a,b in zip(profile[:-1],profile[1:]):
            self.collimator[keys[(a>b).astype(np.int)]] += [x for _ in range(min([a,b]),max([a,b]))]
            x += 1
        self.beamlets = self.intensity*profile[1:-1]

class PlannerInterface:
    def __init__(self, filename):
        self.datafile = filename
        self.horiz_beam = Beam(BeamDirection.Horizontal)
        self.vert_beam = Beam(BeamDirection.Vertical)
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

        :return dose_map: the accumulated dose map at the optimized collimator sequences, of the same shape as the input maps
        :return horiz_beam: the horizontal beam object, with beam intensity, collimator, and beamlets attributes 
        :return vert_beam: the vertical beam object, with beam intensity, collimator, and beamlets attributes 

        See optimize_demo.ipynb for example
        """
        m,n = self.shape

        # Step 1: Optimize intensity profiles (beamlets) for each beam.
        beamlets, dose = self.optimize_beamlets(smoothness, tol, maxiter, bounds)
        
        # Step 2: Compute the beam intensities and sequence of collimator apertures from the optimized beamlets.
        self.solve_beam_collimators(beamlets, exposure_time)
        
        # Collect values for output.
        beamlet_values = [b.value for b in beamlets]
        dose_map = np.array([d.value for d in dose]).reshape(m,n)
        self.dose_map = dose_map
        self.opt = True # set optimization flag
        return dose_map, self.horiz_beam, self.vert_beam

    def optimize_beamlets(self, smoothness, tol, maxiter, bounds):
        m,n = self.shape
        attenuation = np.zeros((m*n,m+n)) # matrix of attenuation factors
        mu = 0.07 # attenuation coefficient in human tissue (brain, lung, blood) for 1 MeV photon energy
                  # (Ref: http://dergipark.gov.tr/download/article-file/131798)
        for i in range(m):
            attenuation[i*n:(i+1)*n,i] = np.exp(-mu*np.arange(n))
        for j in range(n):
            attenuation[j:m*n:n,m+j] = np.exp(-mu*np.arange(m))

        beamlets = np.array([Var() for _ in range(m+n)]) # individual beamlets for optimization problem
        dose = np.dot(attenuation,beamlets) # accumulated dose in each grid cell
        cost = mean_squared_error(dose,np.ravel(self._maps['target'])) \
             + pos_penalty(beamlets,smoothness)
        if bounds:
            cost+=minmax_penalty(np.ravel(self._maps['min'])-dose)
            cost+=minmax_penalty(dose-np.ravel(self._maps['max']))
           
        step, Niter = BFGS(cost,beamlets,np.ones(len(beamlets)),tol=tol,maxiter=maxiter)
        return beamlets, dose

    def solve_beam_collimators(self, beamlets, exposure_time):
        m,n = self.shape
        horiz_beamlets = [b.value for b in beamlets[:m]]
        vert_beamlets = [b.value for b in beamlets[m:]]
        self.horiz_beam.solve(horiz_beamlets, exposure_time)
        self.vert_beam.solve(vert_beamlets, exposure_time)
        beamlet_values = np.hstack([self.horiz_beam.beamlets,self.vert_beam.beamlets]) # set new beamlets incorporating exposure time
        for b,val in zip(beamlets,beamlet_values):
            b.set_value(val)

    def print_summary(self):
        if not self.opt:
            raise Exception("No summary available; plan has not been optimized.")
        else:
            total_dose = np.sum(self.dose_map)
            avg_dose = np.mean(self.dose_map)
            contents = ""
            for beam in [self.horiz_beam, self.vert_beam]:
                contents += beam.name+" intensity: "+"%.2f mW/cm^2"%beam.intensity+"\n"+ \
                            beam.name+" exposure time: "+"%d sec."%len(beam.collimator["left"])+"\n"
            contents += "Total accumulated dose: "+"%.2f Gy"%total_dose+"\n" \
                        "Average dose per unit area: "+"%.2f Gy/cm^2"%avg_dose+"\n"
            print(contents)



