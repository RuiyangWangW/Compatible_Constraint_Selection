import numpy as np
import math

from .utils import wrap_angle

class Safe_Set_Series2D:
    """
    Represents a series of safe sets in 2D space.

    Attributes:
        centroids (list): A list of centroid coordinates for each safe set.
        radii (list): A list of radii for each safe set.
        id (None): The ID of the safe set.
        num_sets (int): The number of safe sets in the series.
    """

    def __init__(self, centroids, radii):
        self.centroids = centroids
        self.radii = radii
        self.id = None
        self.num_sets = len(centroids)

    def return_centroid(self, id):
        """
        Returns the centroid coordinates of the safe set with the given ID.

        Args:
            id (int): The ID of the safe set.

        Returns:
            tuple: The centroid coordinates (x, y) of the safe set.
        """
        return self.centroids[id]

    def return_radius(self, id):
        """
        Returns the radius of the safe set with the given ID.

        Args:
            id (int): The ID of the safe set.

        Returns:
            float: The radius of the safe set.
        """
        return self.radii[id]

def PointsInCircum(r, n=100):
    """
    Generates a series of points on the circumference of a circle.

    Parameters:
    - r (float): The radius of the circle.
    - n (int): The number of points to generate. Default is 100.

    Returns:
    - numpy.ndarray: An array of points on the circumference of the circle.
    """
    Points = [(math.cos(2*math.pi/n*x)*r, math.sin(2*math.pi/n*x)*r) for x in range(0, n+1)]
    return np.array(Points)

 
def PointsInCircum_with_theta(r, n=100):
    """
    Generate a series of points on the circumference of a circle with radius 'r'.

    Parameters:
    - r (float): The radius of the circle.
    - n (int): The number of points to generate. Default is 100.

    Returns:
    - numpy.ndarray: An array of points on the circumference of the circle. Each point is represented as a tuple (x, y, 0, angle).
    """
    Points = [(math.cos(2*math.pi/n*x)*r, math.sin(2*math.pi/n*x)*r, 0, wrap_angle(2*math.pi/n*x+math.pi/2)) for x in range(0, n+1)]
    return np.array(Points)


