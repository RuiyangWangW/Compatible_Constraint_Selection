import numpy as np
from scipy.stats import norm
from .Safe_Set_Series import PointsInCircum
    # Function implementation goes here
def disturb_value(robot, disturbance, disturb_std, disturb_max, f_max_1, f_max_2, scenario_num):
    """
    Calculate the disturbance value based on the given parameters.

    Parameters:
    - robot: The robot object.
    - disturbance: A boolean indicating whether disturbance is enabled.
    - disturb_std: The standard deviation of the disturbance.
    - disturb_max: The maximum disturbance value.
    - f_max_1: The maximum value of f_max_1.
    - f_max_2: The maximum value of f_max_2.
    - scenario_num: The scenario number.

    Returns:
    - u_d_value: A numpy array representing the disturbance value.

    """
    if scenario_num == 1:
        if disturbance and robot.X[1]>4.0 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
            y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0]/f_max_1 * disturb_max
            x_disturb = 0.0
        elif disturbance and robot.X[0]>-0.5 and robot.X[0] < 1.8\
            and robot.X[1] > -2*(disturb_std*0.5) and robot.X[1] < 2*(disturb_std*0.5):
            x_disturb = norm.pdf(robot.X[1], loc=0, scale=disturb_std*0.5)[0]/f_max_2 * disturb_max
            y_disturb = 0.0
        else:
            x_disturb = 0.0
            y_disturb = 0.0

    elif scenario_num == 2:
        if disturbance and robot.X[1]>3.5 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
            y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0]/f_max_1 * disturb_max
            x_disturb = 0.0
        elif disturbance and robot.X[0]>-0.5 and robot.X[0] < 1.8\
            and robot.X[1] > -2*(disturb_std*0.5) and robot.X[1] < 2*(disturb_std*0.5):
            x_disturb = norm.pdf(robot.X[1], loc=0, scale=disturb_std*0.5)[0]/f_max_2 * disturb_max
            y_disturb = 0.0
        else:
            x_disturb = 0.0
            y_disturb = 0.0

    elif scenario_num == 3:
        if robot.type == 'SingleIntegrator2D':
            if disturbance and robot.X[1]>4.0 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
                y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0]/f_max_1 * disturb_max
                x_disturb = 0.0
            elif disturbance and robot.X[0]>-0.5 and robot.X[0] < 1.8\
                and robot.X[1] > -2*(disturb_std*0.5) and robot.X[1] < 2*(disturb_std*0.5):
                x_disturb = norm.pdf(robot.X[1], loc=0, scale=disturb_std*0.5)[0]/f_max_2 * disturb_max
                y_disturb = 0.0
            else:
                x_disturb = 0.0
                y_disturb = 0.0
        else:
            if disturbance and robot.X[1]>4.0 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
                y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0]/f_max_1 * disturb_max
                x_disturb = 0.0
            elif disturbance and robot.X[0]>-1.5 and robot.X[0] < 1.8\
                and robot.X[1] > -2*(disturb_std*0.5) and robot.X[1] < 2*(disturb_std*0.5):
                x_disturb = norm.pdf(robot.X[1], loc=0, scale=disturb_std*0.5)[0]/f_max_2 * disturb_max
                y_disturb = 0.0
            else:
                x_disturb = 0.0
                y_disturb = 0.0
    else:
        y_disturb = 0.0
        x_disturb = 0.0

    u_d_value = np.array([x_disturb, y_disturb]).reshape(2,1)
    
    return u_d_value

def scenario_waypoints(scenario_num, robot_type):
    """
    Generates a list of waypoints based on the given scenario number and robot type.

    Parameters:
    - scenario_num (int): The scenario number.
    - robot_type (str): The type of robot.

    Returns:
    - centroids (list): A list of waypoints.

    Raises:
    - None

    Example usage:
    >>> scenario_waypoints(1, 'SingleIntegrator2D')
    [[4.5, 1.0], [3.0, 3.0], [0, 5.0], [-1.5, 3.0], [0, 0], [-4.0, 0.0], [-4.0, 3.0]]
    """
    if scenario_num == 1:
        num_points = 7
        centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]
        centroids[0][0] = 4.5
        centroids[0][1] = 1.0

        centroids[1][0] = 3.0
        centroids[1][1] = 3.0

        centroids[2][1] = 5.0
        centroids[2][0] = 0

        centroids[3][0] = -1.5
        centroids[3][1] = 3.0

        centroids[4][0] = centroids[2][0]
        centroids[4][1] = 0

        centroids[5][0] = -4.0
        centroids[5][1] = 0.0

        centroids[6][0] = -4.0
        centroids[6][1] = 3.0
    
    elif scenario_num == 2:
        num_points = 9
        centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]
        centroids[0][0] = 4.5
        centroids[0][1] = 1.0

        centroids[1][0] = 3.0
        centroids[1][1] = 3.0

        centroids[2][0] = 2.5
        centroids[2][1] = 5.0

        centroids[3][1] = 5.0
        centroids[3][0] = 0

        centroids[4][0] = -2.5
        centroids[4][1] = 5.0

        centroids[5][0] = -1.5
        centroids[5][1] = 3.0

        centroids[6][0] = centroids[3][0]
        centroids[6][1] = 0

        centroids[7][0] = -4.0
        centroids[7][1] = 0.0

        centroids[8][0] = -4.0
        centroids[8][1] = 3.0

    elif scenario_num == 3 and robot_type == 'SingleIntegrator2D':
        num_points = 15
        centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]

        centroids[0][0] = 4.5
        centroids[0][1] = 1.0

        centroids[1][0] = 3.8
        centroids[1][1] = 2.3

        centroids[2][0] = 3.2
        centroids[2][1] = 3.8

        centroids[3][0] = 2.0
        centroids[3][1] = 5.0

        centroids[4][0] = 0.8
        centroids[4][1] = 5.0

        centroids[5][0] = -0.8
        centroids[5][1] = 5.0

        centroids[6][0] = -2.0
        centroids[6][1] = 5.0

        centroids[7][0] = -1.4
        centroids[7][1] = 4.0

        centroids[8][0] = -0.6
        centroids[8][1] = 2.6

        centroids[9][0] = 0.2
        centroids[9][1] = 1.3

        centroids[10][0] = 0.0
        centroids[10][1] = 0.0

        centroids[11][0] = -1.5
        centroids[11][1] = 0.0

        centroids[12][0] = -3.5
        centroids[12][1] = 1.0

        centroids[13][0] = -4.0
        centroids[13][1] = 2.0

        centroids[14][0] = -4.5
        centroids[14][1] = 3.0

    elif scenario_num == 3 and robot_type == 'DoubleIntegrator2D':
        num_points = 15
        centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]
        centroids[0][0] = 4.5
        centroids[0][1] = 1.0

        centroids[1][0] = 3.8
        centroids[1][1] = 2.3

        centroids[2][0] = 3.2
        centroids[2][1] = 3.8

        centroids[3][0] = 2.0
        centroids[3][1] = 4.8

        centroids[4][0] = 0.7
        centroids[4][1] = 5.2

        centroids[5][0] = -0.7
        centroids[5][1] = 5.2

        centroids[6][0] = -2.0
        centroids[6][1] = 4.8

        centroids[7][0] = -1.2
        centroids[7][1] = 3.8

        centroids[8][0] = -0.6
        centroids[8][1] = 2.6

        centroids[9][0] = 0.2
        centroids[9][1] = 1.5

        centroids[10][0] = -0.75
        centroids[10][1] = 0.0

        centroids[11][0] = -2.5
        centroids[11][1] = 0.0

        centroids[12][0] = -3.5
        centroids[12][1] = 1.5

        centroids[13][0] = -4.0
        centroids[13][1] = 2.5

        centroids[14][0] = -4.5
        centroids[14][1] = 3.5
    else:
        return 0

    return centroids
