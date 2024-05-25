import numpy as np

def wrap_angle(angle):
    """
    Wraps an angle to the range [-pi, pi].

    Parameters:
    angle (float): The angle to be wrapped.

    Returns:
    float: The wrapped angle.

    """
    if angle > np.pi:
        return angle - 2 * np.pi
    elif angle < -np.pi:
        return angle + 2 * np.pi
    else:
        return angle

def euler_to_rot_mat(phi, theta, psi):
    """
    Converts Euler angles to a rotation matrix.

    Parameters:
    phi (float): Rotation angle around the x-axis in radians.
    theta (float): Rotation angle around the y-axis in radians.
    psi (float): Rotation angle around the z-axis in radians.

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    return np.array([[np.cos(psi) * np.cos(theta), -np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi), np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi) * np.sin(theta)],
                     [np.sin(psi) * np.cos(theta), np.cos(psi) * np.cos(phi) + np.sin(psi) * np.sin(theta) * np.sin(phi), -np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi) * np.cos(phi)],
                     [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]])
     
def euler_rate_matrix(phi, theta):
    """
    Calculates the Euler rate matrix given the Euler angles phi and theta.

    Parameters:
    phi (float): The Euler angle phi in radians.
    theta (float): The Euler angle theta in radians.

    Returns:
    numpy.ndarray: The Euler rate matrix as a 3x3 numpy array.
    """
    return np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi) / np.cos(theta), np.cos(phi) * np.sin(theta)]])

    
    