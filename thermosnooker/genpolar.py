"""
This module generates polar coordinates for rings of points, where each ring can have an increasing
number of points based on a multiplicative factor. This is used for setting the initial positions for the balls.
"""
import numpy as np
def rtrings(rmax, nrings, multi):
    """
    Generate polar coordinates for a set of rings with increasing radii and density of points.

    The function/generator yields tuples of polar coordinates (r, theta), starting from the center
    of the coordinate system and moving outward with increasing ring radii and point density.
    
    Args:
        rmax (float): The maximum radius, which is the radius of the outermost ring.
        nrings (int): The number of rings to generate.
        multi (int): The multiplicative factor determining the number of points in each successive ring.
    
    Yields:
        tuple: A tuple (r, theta) representing the radius and angle of a point in polar coordinates.
    """
    yield (0.0, 0.0)  # Yield the center point
    for ring in range(1, nrings + 1):
        points = multi * ring  # Number of points in the current ring increases linearly
        for point in range(points):
            rad = ring / nrings * rmax  # Radius for current ring
            theta = point / points * 2 * np.pi  # Angle for current point in the current ring
            yield (rad, theta)
