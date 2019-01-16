import numpy as np


def dist(s, d):
    return np.abs(s[0]-d[0]) + np.abs(s[1]-d[1])
