
import numpy as np
import matplotlib.pyplot as plt
from gradpy.autodiff import Var
from gradpy.math import *
from .bfgs import BFGS
from .costfunctions import *
from .timer import timer

from enum import Enum


class BeamDirection(Enum):
    """ Enum of beam directions used by therapy planner."""
    Horizontal = 1  # beam propagating along x
    Vertical = 2  # beam propagating along y


class Beam:
    """ Class to compute beam exposure time and collimator aperture sequences. """
    def __init__(self, direction):
        if direction not in [BeamDirection.Horizontal,BeamDirection.Vertical]:
            raise Exception("Invalid beam direction.")
        else:
            self.direction = direction
            self.name = direction.name+" beam"
        self.collimator = None
        self.beamlets = None

    def solve(self, beamlets, intensity):
        """
        Compute beam exposure time and collimator aperture sequence
        based on optimized beamlets and beam intensity.

        INPUTS
        =======
        beamlets: an array of beamlet source intensities.
        intensity: the true single beamlet intensity.

        COMPUTES
        ========
        collimator: a dictionary of the left and right collimator position sequences.
        exposure_time: the corresponding exposure time for the beam.
        """
        self.intensity = intensity
        profile = np.array([0] + [(b//self.intensity+np.round(b%self.intensity)/self.intensity).astype(np.int) \
                                  for b in beamlets] + [0], dtype=np.int)
        keys = ["left", "right"]
        self.collimator = {"left": [], "right": []}
        x = 0
        for a,b in zip(profile[:-1], profile[1:]):
            self.collimator[keys[(a > b).astype(np.int)]] += [x for _ in range(min([a, b]), max([a, b]))]
            x += 1
        self.beamlets = self.intensity*profile[1:-1]
        self.exposure_time = len(self.collimator["left"])


class PlannerInterface:
    def __init__(self, filename):
        """
        Interface for solving and visualizing 2D dose delivery optimization.

        INPUTS
        =======
        filename: The filename of user-defined target, maximum, and minimum dose maps.

        ATTRIBUTES
        ===========
        datafile: the input filename for dose maps.
        opt: flag indicating if optimization has been performed.
        rotate: If rotation of the dose map is performed in optimization.
        rotation_angle: the optimal rotation angle.

        METHODS
        ========
        get_maps(): get dictionary of available dose maps.
        optimize(): optimize dose delivery plan.
        plot_map(): visualize a dose map.
        plot_collimators(): visualize the optimized collimator aperture sequences.
        print_summary(): display a summary of the optimized plan.
        """
        self.datafile = filename
        horiz_beam = Beam(BeamDirection.Horizontal)
        vert_beam = Beam(BeamDirection.Vertical)
        self._beams = [horiz_beam, vert_beam]
        self._maps = self.read_maps()
        out1 = np.zeros_like(self._maps['min'])
        out2 = np.zeros_like(self._maps['min'])
        out3 = np.zeros_like(self._maps['min'])

        if not (self._maps['target'].shape == self._maps['min'].shape and self._maps['target'].shape == self._maps['max'].shape):
            raise Exception('All maps must have the same shape.')
        elif np.any(np.greater(self._maps['min'], self._maps['max'], out=out1, where=((~np.isnan(self._maps['max']))&(~np.isnan(self._maps['min']))))):
            raise Exception("The entries on the minimum map are larger than the ones on the maximum map.")
        elif np.any(np.greater(self._maps['min'], self._maps['target'], out=out2, where=((~np.isnan(self._maps['target']))&(~np.isnan(self._maps['min']))))):
            raise Exception("The entries on the minimum map are larger than the ones on the target map.")
        elif np.any(np.greater(self._maps['target'], self._maps['max'], out=out3, where=((~np.isnan(self._maps['max']))&(~np.isnan(self._maps['target']))))):
            raise Exception("The entries on the target map are larger than the ones on the maximum map.")

        self.opt = False  # flag if plan has been optimized

        # Initialize rotation info
        self.rotate = None
        self.rotation_angle = None # counter-clockwise in degrees

    def read_maps(self):
        """
        Read text input file of dose maps to a dictionary.

        RETURNS
        =======
        maps: dictionary of maps, where key is "target", "max" or "min" depending
              on the map type and value is an numpy 2D array.
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
                values = [float(x) if x.strip() != 'N' else np.NaN for x in line.split(',')]
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
        return maps

    def get_maps(self):
        return self._maps

    @timer
    def optimize(self, intensity, smoothness=1., tol=1e-8, maxiter=1000, bounds=False, allow_rotation=False):
        """
        Perform 2D dose delivery optimization.

        INPUTS
        =======
        intensity: The intensity of the horizontal and vertical irradiation beams (assumed equal).
        smoothness (optional, default 1): The smoothness of the logistic function used to enforce minimum and maximum constrants.
        tol (optional, default 1e-8): the stopping tolerance for the 2-norm of the step size.
        maxiter (optional, default 100000): the maximum number of iterations.
        bounds (optional, default False): whether to enforce minimum and maximum dose constraints.
        allow_rotation (optional, default False): whether to allow rotation of the maps in optimization.

        COMPUTES
        =========
        "optimized", "difference", and "error" dose maps added to the maps dictionary.
        horizontal and vertical beam exposure times and collimator aperture sequences.

        Results may be visualized with the plot_map(), plot_collimators(), and print_summary() methods.
        """
        if not allow_rotation:
            maps, cost_value, beams = self.optimize_map(self._maps, intensity, smoothness, tol, maxiter, bounds)
            self._maps = maps
            self._beams = beams
        else:
            best_index = None
            best_cost = None
            current_maps = self._maps
            for i in range(4):
                print("Optimize for rotation: {} degrees".format(90*i))
                try:
                    solved_maps, cost_value, beams = self.optimize_map(current_maps,
                                                                   intensity,
                                                                   smoothness, tol,
                                                                   maxiter, bounds)
                    if best_cost is None or cost_value < best_cost:
                        best_cost = cost_value
                        best_index = i
                        self._maps = solved_maps
                        self._beams = beams
                except Exception as e:
                    print(e)
                current_maps = {k: np.rot90(v) for k, v in current_maps.items()}
            if best_index is None:
                raise Exception("Could not find orientation without violating constraints.")
            self.rotation_angle = 90 * best_index
            print("Found best rotation (counter-clockwise): {} degrees".format(self.rotation_angle))
        self.rotate = allow_rotation
        self.opt = True  # set optimization flag

    def optimize_map(self, maps, intensity, smoothness, tol, maxiter, bounds):
        """
        Optimize a single dose map (internal subroutine)

        """
        m, n = maps['target'].shape

        # Step 1: Optimize intensity profiles (beamlets) for each beam.
        beamlets, dose, found, cost_value = self.optimize_beamlets(maps, smoothness, tol, maxiter, bounds)

        # Handle the case when a minimum to the cost functions cannot be found by BFGS
        if found is False:
            raise Exception("The minimum cannot be found given the cost functions and the number of iterations.")

        # Step 2: Compute the beam exposure times and sequence of collimator apertures from the optimized beamlets.
        horiz_beam = Beam(BeamDirection.Horizontal)
        vert_beam = Beam(BeamDirection.Vertical)
        beams = [horiz_beam, vert_beam]
        beams = self.solve_beam_collimators(beams, (m, n), beamlets, intensity)

        # Collect values for output.
        dose_map = np.array([d.value for d in dose]).reshape(m, n)

        if bounds:
            # Check whether the minimum at the cost functions meets the constraints
            out = np.zeros_like(dose_map)
            if np.any(np.greater(dose_map, maps['max'], out=out, where=~np.isnan(maps['max']))):
                raise Exception("The maximum constraints are violated. Suggestion: Adjust the smoothness.")
            out = np.zeros_like(dose_map)
            if np.any(np.less(dose_map, maps['min'], out=out, where=~np.isnan(maps['min']))):
                raise Exception("The minimum constraints are violated. Suggestion: Adjust the smoothness.")

        maps["optimized"] = dose_map
        maps["difference"] = dose_map - maps["target"]  # the difference dose map - target map
        maps["error"] = np.abs(maps["difference"])  # magnitude of difference map
        return maps, cost_value, beams

    def optimize_beamlets(self, maps, smoothness, tol, maxiter, bounds):
        """
        Optimize beamlets according to reframed problem (internal subroutine)

        """
        m, n = maps['target'].shape
        attenuation = np.zeros((m*n, m+n)) # matrix of attenuation factors
        mu = 0.07  # attenuation coefficient in human tissue (brain, lung, blood) for 1 MeV photon energy
                   # (Ref: http://dergipark.gov.tr/download/article-file/131798)
        for i in range(m):
            attenuation[i*n:(i+1)*n, i] = np.exp(-mu*np.arange(n))
        for j in range(n):
            attenuation[j:m*n:n, m+j] = np.exp(-mu*np.arange(m))

        beamlets = np.array([Var() for _ in range(m+n)])  # individual beamlets for optimization problem
        dose = np.dot(attenuation, beamlets)  # accumulated dose in each grid cell
        cost = mean_squared_error(dose, np.ravel(maps['target'])) \
             + pos_penalty(beamlets, smoothness)
        if bounds:
            cost += minmax_penalty(dose, np.ravel(maps['max']), smoothness)
            cost += minmax_penalty(-dose, -np.ravel(maps['min']), smoothness)

        step, Niter, found = BFGS(cost, beamlets, np.ones(len(beamlets)), tol=tol, maxiter=maxiter)
        return beamlets, dose, found, cost.value

    def solve_beam_collimators(self, beams, map_shape, beamlets, intensity):
        """
        Solve for beam collimator apertures and exposure times (internal subroutine)

        """
        m, n = map_shape
        horiz_beamlets = [b.value for b in beamlets[:m]]
        vert_beamlets = [b.value for b in beamlets[m:]]
        if np.any(np.array(horiz_beamlets)<0) or np.any(np.array(vert_beamlets)<0):
            raise Exception("Negative beamlet value detected. Suggestion: Adjust the smoothness.")
        beams[0].solve(horiz_beamlets, intensity)
        beams[1].solve(vert_beamlets, intensity)
        beamlet_values = np.hstack([beams[0].beamlets, beams[1].beamlets])  # set new beamlets incorporating intensity
        for b, val in zip(beamlets, beamlet_values):
            b.set_value(val)
        return beams

    def plot_map(self, name, ax=None, cmap='viridis_r', fontsize=14):
        """
        Plot a dose map for visualization.

        INPUTS
        =======
        name: The name of map to plot from the dictionary in get_maps(); before optimization,
              valid names are "target", "min", and "max";
              after optimization, additional maps are "optimized", "difference", and "error".
        ax (optional, default None) ax: the axes to which the plot should be added. If none, plot_map creates a figure and axes.
        cmap (optional, default 'viridis_r'): the matplotlib colormapp to use.
        fontsize (optional, default 14): the font size for axis labels and text.
        """
        try:
            dose_map = self._maps[name]
        except KeyError:
            raise Exception('Map "'+name+'" does not exist.')
        m, n = dose_map.shape
        if ax is None:
            show = True
            fig, ax = plt.subplots(1, 1, figsize=(n+m/2, m))
        else:
            show = False
        im = ax.imshow(dose_map, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("dose (Gy)", rotation=-90, va="bottom", size=14)
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        ax.set_xticks([])
        ax.set_yticks([])
        textcolors=["black", "white"]
        threshold = im.norm(np.nanmax(dose_map))/2.
        for i in range(m):
            for j in range(n):
                text = ax.text(j, i, np.round(dose_map[i,j],2), size=fontsize,
                               ha="center", va="center", color=textcolors[(im.norm(dose_map[i,j])>threshold).astype(np.int)])
        ax.text(0.5, -0.1, "{} map".format(name),
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, size=14)
        if show:
            plt.show()

    def plot_collimators(self):
        """
        Plot optimized collimator aperture sequences.
        """
        if not self.opt:
            raise Exception("Collimator sequences not available; plan has not been optimized.")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            m, n = self._maps["target"].shape
            for t, (left, right) in enumerate(zip(self._beams[0].collimator["left"], self._beams[0].collimator["right"])):
                ax1.plot([t, t], [m+1, m-left], lw=10, color=[0.2, 0.6, 0.7])
                ax1.plot([t, t], [m-right, -1], lw=10, color=[0.2, 0.6, 0.7])
            ax1.axhline(0, linestyle='dashed', color='k')
            ax1.axhline(m, linestyle='dashed', color='k')
            ax1.axhspan(0, m, alpha=0.5, color=[1.0,1.0,0.1])
            ax1.set_xlim(-0.5, len(self._beams[0].collimator["left"]))
            ax1.tick_params(labelsize=14)
            ax1.set_xlabel('exposure time (s)', size=14)
            ax1.set_title('horizontal beam collimator apertures', size=14)
            for t, (left, right) in enumerate(zip(self._beams[1].collimator["left"], self._beams[1].collimator["right"])):
                ax2.plot([-1, left], [t, t], lw=10, color=[0.2, 0.6, 0.7])
                ax2.plot([right, n+1], [t, t], lw=10, color=[0.2, 0.6, 0.7])
            ax2.axvline(0, linestyle='dashed', color='k')
            ax2.axvline(n, linestyle='dashed', color='k')
            ax2.axvspan(0, n, alpha=0.5, color=[1.0, 1.0, 0.1])
            ax2.set_ylim(-0.5, len(self._beams[1].collimator["left"]))
            ax2.tick_params(labelsize=14)
            ax2.set_ylabel('exposure time (s)', size=14)
            ax2.set_title('vertical beam collimator apertures', size=14)
            plt.show()

    def print_summary(self):
        """
        Print a summary of the optimized plan.
        """
        if not self.opt:
            raise Exception("No summary available; plan has not been optimized.")
        else:
            if self.rotate is True:
                print("Maps are rotated for optimality.")
                print("Optimal rotation (counter-clockwise): {} degrees".format(self.rotation_angle))
            total_dose = np.sum(self._maps["optimized"])
            avg_dose = np.mean(self._maps["optimized"])
            contents = ""
            for beam in self._beams:
                contents += beam.name+" intensity: "+"%.2f mW/cm^2"%beam.intensity+"\n"+ \
                            beam.name+" exposure time: "+"%d sec."%beam.exposure_time+"\n"
            contents += "Total accumulated dose: "+"%.2f Gy"%total_dose+"\n" \
                        "Average dose per unit area: "+"%.2f Gy/cm^2"%avg_dose+"\n"
            print(contents)
