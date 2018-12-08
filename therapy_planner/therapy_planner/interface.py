import sys
sys.path.append("../../gradpy")
import numpy as np
import matplotlib.pyplot as plt
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

    def solve(self, beamlets, intensity):
        """
        Compute beam exposure time and collimator aperture sequence
        based on optimized beamlets and beam intensity.
        """
        self.intensity = intensity
        profile = np.array([0] + [(b//self.intensity+np.round(b%self.intensity)/self.intensity).astype(np.int) \
                  for b in beamlets] + [0],dtype=np.int)
        keys = ["left","right"]
        self.collimator = {"left": [],"right": []}
        x = 0
        for a,b in zip(profile[:-1],profile[1:]):
            self.collimator[keys[(a>b).astype(np.int)]] += [x for _ in range(min([a,b]),max([a,b]))]
            x += 1
        self.beamlets = self.intensity*profile[1:-1]
        self.exposure_time = len(self.collimator["left"])

class PlannerInterface:
    def __init__(self, filename):
        self.datafile = filename
        horiz_beam = Beam(BeamDirection.Horizontal)
        vert_beam = Beam(BeamDirection.Vertical)
        self._beams = [horiz_beam, vert_beam]
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

    def optimize(self, intensity, smoothness=1., tol=1e-8, maxiter=1000, bounds=False):
        """
        :param intensity: the intensity of the incident beams
        :param smoothness: the smoothness of the logistic function used in the penalty term
        :param tol: stopping criterion for step size in optimization
        :param maxiter: maximum number of iterations in optimization

        :computes dose_map: the accumulated dose map at the optimized collimator sequences, of the same shape as the input maps
        :computes horiz_beam: the horizontal beam object, with beam intensity, collimator, and beamlets attributes 
        :computes vert_beam: the vertical beam object, with beam intensity, collimator, and beamlets attributes 

        See optimize_demo.ipynb for example
        """
        m,n = self.shape

        # Step 1: Optimize intensity profiles (beamlets) for each beam.
        beamlets, dose, found = self.optimize_beamlets(smoothness, tol, maxiter, bounds)

        # Handle the case when a minimum to the cost functions cannot be found by BFGS
        if found is False:
            print("The minimum cannot be found given the cost functions and the number of iterations.")
            return None

        # Step 2: Compute the beam exposure times and sequence of collimator apertures from the optimized beamlets.
        self.solve_beam_collimators(beamlets, intensity)
        
        # Collect values for output.
        beamlet_values = [b.value for b in beamlets]
        dose_map = np.array([d.value for d in dose]).reshape(m,n)

        # Check whether the minimum at the cost functions meets the constraints, if bounds are desired.
        if bounds:
            v = False
            if np.sum(dose_map > self._maps['max']) != 0:
                print("The maximum constraints are violated. Suggestion: Adjust the smoothness.")
                v = True
            if np.sum(dose_map < self._maps['min']) != 0:
                print("The minimum constraints are violated. Suggestion: Adjust the smoothness.")
                v = True
            if v is True:
                return None

        self._maps["optimized"] = dose_map
        self._maps["difference"] = dose_map - self._maps["target"] # the difference dose map - target map
        self._maps["error"] = np.abs(self._maps["difference"]) # magnitude of difference map
        self.opt = True # set optimization flag

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
            cost+=minmax_penalty(np.ravel(self._maps['min'])-dose, smoothness)
            cost+=minmax_penalty(dose-np.ravel(self._maps['max']), smoothness)
           
        step, Niter, found = BFGS(cost,beamlets,np.ones(len(beamlets)),tol=tol,maxiter=maxiter)
        return beamlets, dose, found

    def solve_beam_collimators(self, beamlets, intensity):
        m,n = self.shape
        horiz_beamlets = [b.value for b in beamlets[:m]]
        vert_beamlets = [b.value for b in beamlets[m:]]
        self._beams[0].solve(horiz_beamlets, intensity)
        self._beams[1].solve(vert_beamlets, intensity)
        beamlet_values = np.hstack([self._beams[0].beamlets,self._beams[1].beamlets]) # set new beamlets incorporating intensity
        for b,val in zip(beamlets,beamlet_values):
            b.set_value(val)

    def plot_map(self, name, ax=None, cmap='viridis_r', fontsize=14):
        """
        param name: name of map to plot; before optimization, valid names are "target", "min", and "max"
                                         after optimization, additional maps are "optimized", "difference", and "error".
        param ax: the axes to which the plot should be added.
        param cmap: the colormap to use (default 'viridis_r')
        param fontsize: the font size for axis labels and text (default 14)
        """
        try:
            dose_map = self._maps[name]
        except KeyError:
            raise Exception('Map "'+name+'" does not exist.')
        m,n = dose_map.shape
        if ax is None:
            show = True
            fig, ax = plt.subplots(1,1,figsize=(n+m/2,m))
        else:
            show = False
        im = ax.imshow(dose_map, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("dose", rotation=-90, va="bottom", size=14)
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        ax.set_xticks([])
        ax.set_yticks([])
        textcolors=["black", "white"]
        threshold = im.norm(dose_map.max())/2.
        for i in range(m):
            for j in range(n):
                text = ax.text(j,i,np.round(dose_map[i,j],2),size=fontsize,
                               ha="center", va="center",color=textcolors[(im.norm(dose_map[i,j])>threshold).astype(np.int)])
        ax.text(0.5,-0.1,"{} map".format(name),
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, size=14)
        if show:
            plt.show()

    def plot_collimators(self):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
        m,n = self.shape
        for t,(left, right) in enumerate(zip(self._beams[0].collimator["left"],self._beams[0].collimator["right"])):
            ax1.plot([t,t],[m+1,m-left],lw=10,color=[0.2,0.6,0.7])
            ax1.plot([t,t],[m-right,-1],lw=10,color=[0.2,0.6,0.7])
        ax1.axhline(0,linestyle='dashed',color='k')
        ax1.axhline(m,linestyle='dashed',color='k')
        ax1.axhspan(0,m,alpha=0.5,color=[1.0,1.0,0.1])
        ax1.set_xlim(-0.1,len(self._beams[0].collimator["left"]))
        ax1.set_xlabel('exposure time',size=14)
        ax1.set_title('horizontal beam collimator apertures',size=14)
        for t,(left, right) in enumerate(zip(self._beams[1].collimator["left"],self._beams[1].collimator["right"])):
            ax2.plot([-1,left],[t,t],lw=10,color=[0.2,0.6,0.7])
            ax2.plot([right,n+1],[t,t],lw=10,color=[0.2,0.6,0.7])
        ax2.axvline(0,linestyle='dashed',color='k')
        ax2.axvline(n,linestyle='dashed',color='k')
        ax2.axvspan(0,n,alpha=0.5,color=[1.0,1.0,0.1])
        ax2.set_ylim(-0.1,len(self._beams[1].collimator["left"]))
        ax2.set_ylabel('exposure time',size=14)
        ax2.set_title('vertical beam collimator apertures',size=14)
        plt.show()

    def print_summary(self):
        if not self.opt:
            raise Exception("No summary available; plan has not been optimized.")
        else:
            total_dose = np.sum(self._maps["optimized"])
            avg_dose = np.mean(self._maps["optimized"])
            contents = ""
            for beam in self._beams:
                contents += beam.name+" intensity: "+"%.2f mW/cm^2"%beam.intensity+"\n"+ \
                            beam.name+" exposure time: "+"%d sec."%beam.exposure_time+"\n"
            contents += "Total accumulated dose: "+"%.2f Gy"%total_dose+"\n" \
                        "Average dose per unit area: "+"%.2f Gy/cm^2"%avg_dose+"\n"
            print(contents)



