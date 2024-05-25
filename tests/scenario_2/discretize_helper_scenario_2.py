import numpy as np
from scipy.stats import norm
import math
import sys
sys.path.append("../../src")
from robot_models.SingleIntegrator2D import *

def discretize_u_forward_cal(x0):
    """
    Discretizes the forward set of a robot given an initial position.

    Parameters:
    - x0 (numpy.ndarray): The initial position of the robot.

    Returns:
    - x0_key (str): The key representing the initial position in the grid.
    - forward_set (numpy.ndarray): The forward set of the robot, represented as an array of positions.
    - dist_list (numpy.ndarray): The distances between each position in the forward set and the initial position.
    """
    
    #Define Constants
    dt = 0.2
    U_max = 1.0

    #Define Disturbance
    disturbance = True
    disturb_max = 1.5*U_max
    disturb_std = 1.5
    f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
    f_max_2 = f_max_1/0.5

    #Define Grid
    y_max = 6.0
    y_min = -2.0
    x_min = -6.0
    x_max = 6
    step = 0.1
    
    # Define x and y range
    x_range = np.arange(start=x_min, stop=x_max+step, step=step)
    y_range = np.arange(start=y_min, stop=y_max+step, step=step)

    # Define u_list
    u_step = 0.1
    u_list = np.arange(start=-U_max,stop=U_max+u_step,step=u_step)
    u2d_list = np.zeros(shape=(u_list.shape[0]**2,2))
    for i in range(u_list.shape[0]):
        for j in range(u_list.shape[0]):
            if u_list[i]==0 and u_list[j]==0:
                continue
            u = np.array([u_list[i],u_list[j]])
            if (np.linalg.norm(u)>U_max):
                u /= np.linalg.norm(u)
                u *= U_max
            u2d_list[u_list.shape[0]*i+j,:] = u.reshape(-1,2)

    robot = SingleIntegrator2D(x0, dt, ax=None, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=False)
    forward_set = np.array([])

    dist_list = np.array([])
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
    
    u_disturb = np.array([x_disturb, y_disturb]).reshape(2,1)

    x0_key = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))

    has_been_added = {}

    # Loop through all possible u values
    for i in range(u2d_list.shape[0]):
        robot.X = x0.reshape(-1,1)
        u = u2d_list[i,:].reshape(2,1)
        u_next = u + u_disturb
        robot.step(u_next)
        new_pos = robot.X
        x = new_pos[0]
        y = new_pos[1]
        if y > y_max or y < y_min or x > x_max or x < x_min:
            continue
        if x > -3.0 and x < -0.6 and y < 1.4 and y > 1.0:
            continue
        if x > 1.0 and x < 2.0 and y < 1.4 and y > -1.0:
            continue
        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
        added = has_been_added.get(pos_key)
        if added:
            continue
        has_been_added.update({pos_key: True})
        forward_set = np.append(forward_set,np.array([pos_key]),axis=0)
        x = x_range[int((x-x_min)/step)]
        y = y_range[int((y-y_min)/step)]
        dist = np.sqrt((x-x0[0])**2+(y-x0[1])**2)
        
        if np.size(dist_list) == 0:
            dist_list = np.array([dist]).reshape(-1,1) 
        else:
            dist_list = np.append(dist_list,np.array([dist]))

    
    return x0_key, forward_set, dist_list