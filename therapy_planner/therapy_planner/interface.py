import numpy as np


def read_maps(filename):
    """
    :param filename: The filename of the text file where user defined the therapy maps (as in the format of the demo.map)
    :return: maps, where key is "target", "max" or "min" depending on the map tyoe and value is an numpy 2D array

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
