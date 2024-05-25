import numpy as np
import cvxpy as cp
import random
import math
import copy

from .utils.scenario_disturb import *
from .robot_models.SingleIntegrator2D import *
from .robot_models.DoubleIntegrator2D import *


class predictive_frame_lag:
    """
    A class that represents a predictive frame lag controller.

    Parameters:
    - scenario_num (int): The scenario number.
    - robot_type (str): The type of robot ('SingleIntegrator2D' or 'DoubleIntegrator2D').
    - x0 (numpy.ndarray): The initial state of the robot.
    - dt (float): The time step.
    - tf (float): The final time.
    - U_max (float): The maximum control signal.
    - V_max (float): The maximum speed.
    - alpha_values (list): The alpha values.
    - beta_values (list): The beta values.
    - num_constraints_hard (int): The number of hard constraints.
    - x_r_list (list): The list of waypoints.
    - radius_list (list): The list of radii.
    - reward_list (list): The list of rewards.
    - obstacle_list (numpy.ndarray): The list of obstacle states.
    - disturbance (float): The disturbance value.
    - disturb_std (float): The disturbance standard deviation.
    - disturb_max (float): The maximum disturbance value.
    """

    def __init__(self, scenario_num, robot_type, x0, dt, tf, U_max, V_max, alpha_values, beta_values, num_constraints_hard, x_r_list, 
                 radius_list, reward_list, obstacle_list, disturbance, disturb_std, disturb_max):
        """
        Initializes the predictive_frame_lag object.

        Parameters:
        - scenario_num (int): The scenario number.
        - robot_type (str): The type of robot ('SingleIntegrator2D' or 'DoubleIntegrator2D').
        - x0 (numpy.ndarray): The initial state of the robot.
        - dt (float): The time step.
        - tf (float): The final time.
        - U_max (float): The maximum control signal.
        - V_max (float): The maximum speed.
        - alpha_values (list): The alpha values.
        - beta_values (list): The beta values.
        - num_constraints_hard (int): The number of hard constraints.
        - x_r_list (list): The list of waypoints.
        - radius_list (list): The list of radii.
        - reward_list (list): The list of rewards.
        - obstacle_list (numpy.ndarray): The list of obstacle states.
        - disturbance (float): The disturbance value.
        - disturb_std (float): The disturbance standard deviation.
        - disturb_max (float): The maximum disturbance value.
        """

        # Initialize the attributes
        self.scenario_num = scenario_num
        self.x0 = x0
        self.dt = dt
        self.tf = tf
        self.num_steps = int(self.tf/self.dt)
        self.U_max = U_max
        self.num_constraints_hard = num_constraints_hard
        self.num_constraints_soft = 1
        self.obstacle_list = obstacle_list
        self.reward_list = reward_list
        self.robot_type = robot_type

        # Create the robot object based on the robot type
        if self.robot_type == 'SingleIntegrator2D':
            self.robot = SingleIntegrator2D(self.x0, self.dt, ax=None, id=0, color='r', palpha=1.0, 
                                            num_constraints_hard=self.num_constraints_hard,
                                            num_constraints_soft=self.num_constraints_soft, plot=False)
        else:
            self.robot = DoubleIntegrator2D(self.x0, self.dt, ax=None, V_max=V_max,
                                            num_constraints_hard=self.num_constraints_hard,
                                            num_constraints_soft=self.num_constraints_soft, plot=False)

        self.disturbance = disturbance
        self.disturb_std = disturb_std
        self.disturb_max = disturb_max
        self.f_max_1 = 1/(self.disturb_std*math.sqrt(2*math.pi))
        self.f_max_2 = self.f_max_1*2.0
        self.x_r_list = x_r_list
        self.radius_list = radius_list
        self.x_r_id = 0
        self.beta_1 = beta_values[0]
        self.beta_2 = beta_values[1]
        self.alpha_1 = alpha_values[0]
        self.alpha_2 = alpha_values[1]
        self.y_max = 6.0
        self.t_limit = self.tf

    def forward(self):
        """
        Executes the forward pass of the predictive_frame_lag module.

        Returns:
            x_list (list): List of x-coordinates of the robot's position at each time step.
            y_list (list): List of y-coordinates of the robot's position at each time step.
            t_list (list): List of time values at each time step.
            lamda_sum_list (list): List of cumulative dual variable values at each time step.
            flag (str): Indicates the success or failure of the optimization problem.
            r (float): Cumulative sum of the dual variable values.
            reward (float): Cumulative sum of rewards obtained during the optimization.
        """
        
        if self.robot_type == 'SingleIntegrator2D':
          # Define constrained Optimization Problem
          u1 = cp.Variable((2,1))
          u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
          alpha_soft = cp.Variable((self.num_constraints_soft))
          alpha_0 = cp.Parameter((self.num_constraints_soft))
          alpha_0.value = np.array([self.alpha_1])
          h = cp.Parameter((self.num_constraints_soft))
          A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
          b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
          A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
          b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))

          const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + cp.multiply(alpha_soft, h), \
                    cp.norm2(u1) <= self.U_max,
                    alpha_soft >= np.zeros((self.num_constraints_soft))]
          objective1 = cp.Minimize(cp.sum_squares(u1 - u1_ref) + 10*cp.sum_squares(alpha_soft-alpha_0))
          constrained_controller = cp.Problem(objective1, const1)
        else:
          # Define constrained Optimization Problem
          u1 = cp.Variable((2,1))
          u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
          A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
          b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
          A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
          b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))
          alpha_soft = cp.Variable((self.num_constraints_soft,1))
          alpha_1_param = cp.Parameter((self.num_constraints_soft,1))
          alpha_2_param = cp.Parameter((self.num_constraints_soft,1))
          alpha_1_param.value = np.array([self.alpha_1]).reshape(1,1)
          alpha_2_param.value = np.array([self.alpha_2]).reshape(1,1)
          phi = cp.Parameter((self.num_constraints_soft,1))
          dphi_dx_T = cp.Parameter((self.num_constraints_soft,2))
          dx12_dt = cp.Parameter((2,self.num_constraints_soft))
          const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + cp.multiply(alpha_soft+alpha_2_param,dphi_dx_T@dx12_dt) +
                         cp.multiply(cp.multiply(alpha_soft,alpha_2_param),phi),
                         cp.norm2(u1[0]) <= self.U_max, cp.norm2(u1[1]) <= self.U_max]
          objective1 = cp.Minimize(cp.sum_squares(u1-u1_ref) + 10*cp.sum_squares(alpha_soft-alpha_1_param))
          constrained_controller = cp.Problem(objective1, const1)  
        
        robot = self.robot
        # Define Disturbance 
        u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

        # Define Reward
        r = 0
        reward = 0
        # Define curr t
        curr_t = 0
        t = 0
        lamda_sum = 0
        lamda_sum_list = []
        flag = "success"
        x_list = []
        y_list = []
        t_list = []

        
        for i in range(self.num_steps):
            
            u_d.value = disturb_value(robot, self.disturbance, self.disturb_std, self.disturb_max, self.f_max_1, self.f_max_2, scenario_num=self.scenario_num)
            x_r = self.x_r_list[self.x_r_id].reshape(2,1)
            radius = self.radius_list[self.x_r_id]

            if self.robot_type == 'SingleIntegrator2D': 
              # Define CBF constraints      
              h1, dh1_dx = robot.static_safe_set(x_r,radius)    
              robot.A1_soft[0,:] = -dh1_dx@robot.g()
              robot.b1_soft[0] = dh1_dx@robot.f() + dh1_dx@robot.g()@u_d.value
              h.value = np.array([h1])

              for j in range(0,len(self.obstacle_list)):
                obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                h_obs, dh_obs_dx = robot.static_safe_set(obs_x_r,0.2) 
                h_obs = -h_obs
                dh_obs_dx = -dh_obs_dx
                robot.A1_hard[j,:] = -dh_obs_dx@robot.g()
                robot.b1_hard[j] = dh_obs_dx@robot.f() + self.beta_1*h_obs + dh_obs_dx@robot.g()@u_d.value
            
            else:
              # Define CBF constraints
              phi_0, dphi_0_dx, dx12_dt_0 = robot.lyapunov(x_r)
              phi.value = np.array([phi_0]).reshape(-1,1)
              dphi_dx_T.value = dphi_0_dx.T
              dx12_dt.value = dx12_dt_0
              robot.A1_soft[0,:] = -dphi_0_dx.T@robot.J()
              robot.b1_soft[0] = dphi_0_dx.T@u_d.value - 2*dx12_dt_0.T@dx12_dt_0
              h1, _, _ = robot.barrier(x_r, radius)
              h1 = -h1

              for j in range(0,len(self.obstacle_list)):
                  obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                  h, dh_dx, dx12_dt_0 = robot.barrier(obs_x_r,0.2)
                  robot.A1_hard[j,:] = -dh_dx.T@robot.J()
                  robot.b1_hard[j] = dh_dx.T@u_d.value + 2*dx12_dt_0.T@dx12_dt_0 + \
                  (self.beta_1+self.beta_2)*dh_dx.T@dx12_dt_0 + (self.beta_1*self.beta_2)*h
                
            A1_soft.value = robot.A1_soft.reshape(-1,2)
            b1_soft.value = robot.b1_soft.reshape(-1,1)
            A1_hard.value = robot.A1_hard.reshape(-1,2)
            b1_hard.value = robot.b1_hard.reshape(-1,1)
            u1_ref.value = robot.nominal_input(x_r)
            
            try:
              constrained_controller.solve()
              lamda_sum += const1[1].dual_value[0][0]
              if robot.type == 'SingleIntegrator2D':
                u_next = u1.value + u_d.value
                robot.step(u_next)
              else:
                robot.step(u1.value, u_d.value, self.disturbance)
            except Exception as error:
              #print(error)
              flag = "fail"

            if constrained_controller.status != "optimal" and constrained_controller.status != "optimal_inaccurate":
              flag = "fail"

            curr_t += self.dt
            x_list.append(robot.X[0])
            y_list.append(robot.X[1])
            t += self.dt
            t_list.append(t)

            if curr_t > self.t_limit:
              flag = "fail"

            if flag == "fail":
              lamda_sum_i = copy.deepcopy(lamda_sum)
              lamda_sum_list.append(lamda_sum_i)
              reward = 0
              break

            if (h1 >= 0):
              lamda_sum_i = copy.deepcopy(lamda_sum)
              lamda_sum = 0
              lamda_sum_list.append(lamda_sum_i)
              if self.x_r_id == len(self.x_r_list)-1:
                break
              reward += self.reward_list[self.x_r_id]
              self.x_r_id += 1
            else:
              continue
        if flag == "success":
          r = 0
        else:
          r = sum(lamda_sum_list)

        return x_list, y_list, t_list, lamda_sum_list, flag, r, reward


def rand_list_init(num_states):
    """
    Initialize a list of random values.

    Parameters:
    num_states (int): The number of states in the list.

    Returns:
    list: A list of random values.

    """
    l = np.ones(shape=(num_states,))
    for i in range(num_states):
        l[i] = random.randint(0,1)
    l = l.tolist()
    return l

def fitness_score_lag(comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, \
                    beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max,\
                    num_constraints_hard, fitness_score_table, mode):
    """
    Calculates the fitness score for a given combination of waypoints.

    Args:
        comb (list): A list representing the combination of waypoints.
        scenario_num (int): The scenario number.
        robot_type (str): The type of robot.
        x0 (float): The initial position of the robot.
        time_horizon (float): The time horizon for prediction.
        reward_max (float): The maximum reward value.
        x_r_list (list): A list of x positions of the waypoints.
        radius_list (list): A list of radii of the waypoints.
        alpha_values (list): A list of alpha values.
        beta_values (list): A list of beta values.
        reward_list (list): A list of rewards for the waypoints.
        U_max (float): The maximum control signal.
        V_max (float): The maximum speed.
        obstacle_list (list): A list of obstacles.
        dt (float): The time step.
        disturbance (bool): A flag indicating if disturbance is present.
        disturb_std (float): The standard deviation of the disturbance.
        disturb_max (float): The maximum disturbance value.
        num_constraints_hard (int): The number of hard constraints.
        fitness_score_table (dict): A dictionary to store fitness scores.

    Returns:
        tuple: A tuple containing the trajectory, lambda sum list, score, reward, and updated fitness score table.
    """

    if fitness_score_table.get(tuple(comb)) != None:
        traj, score, reward = fitness_score_table.get(tuple(comb))
        return traj, [], score, reward, fitness_score_table

    num_states = len(x_r_list)    
    reward_weight = 0.01
    x_r_list_comb = []
    radius_list_comb = []
    reward_list_comb = []
    for i in range(num_states):
        if comb[i] == 1:
            x_r_list_comb.append(x_r_list[i])
            radius_list_comb.append(radius_list[i])
            reward_list_comb.append(reward_list[i])

    if (len(x_r_list_comb)>0):
        pred_frame = predictive_frame_lag(scenario_num,robot_type,x0,dt,time_horizon,U_max,V_max,alpha_values,\
                                          beta_values,num_constraints_hard=num_constraints_hard, \
                                          x_r_list=x_r_list_comb, radius_list=radius_list_comb, \
                                          reward_list = reward_list_comb, obstacle_list=obstacle_list,\
                                          disturbance=disturbance, disturb_std=disturb_std, disturb_max=disturb_max)
        x_list, y_list, t_list, lamda_sum_list, flag, score, reward = pred_frame.forward()
        if flag == "success":
            traj = {"x": x_list, "y": y_list, "t": t_list}
        else:
            traj = {}
    else:
        reward = 0
        traj = {}
        lamda_sum_list = []
    
    if mode != 'deterministic':
        score += (reward_max-reward)*reward_weight
    
    fitness_score_table.update({tuple(comb): [traj, score, reward]})

    return traj, lamda_sum_list, score, reward, fitness_score_table

def mutate_process(comb, mutation_rate):
    """
    Mutates a given combination of states based on a mutation rate.

    Parameters:
    comb (list): The combination of states to be mutated.
    mutation_rate (float): The probability of each state being mutated.

    Returns:
    list: The mutated combination of states.
    """
    mutated_comb = []
    num_states = len(comb)
    for i in range(num_states):
        mutate = np.random.choice([True, False], p=[mutation_rate,1-mutation_rate])
        if mutate:
            mutated_comb.append(random.randint(0,1))
        else:
            mutated_comb.append(comb[i])
    return mutated_comb

def genetic_comb_lag(scenario_num, robot_type, x0, x_r_list, time_horizon, reward_max, radius_list, alpha_values, beta_values, reward_list, U_max, V_max, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):
    """
    Performs genetic combination lag optimization to find the best combination of waypoints.

    Args:
        scenario_num (int): The scenario number.
        robot_type (str): The type of robot.
        x0 (float): The initial position.
        x_r_list (list): The list of waypoints.
        time_horizon (float): The time horizon.
        reward_max (float): The maximum reward.
        radius_list (list): The list of radii.
        alpha_values (list): The list of alpha values.
        beta_values (list): The list of beta values.
        reward_list (list): The list of rewards.
        U_max (float): The maximum control signal.
        V_max (float): The maximum speed.
        obstacle_list (list): The list of obstacles.
        dt (float): The time step.
        disturbance (float): The disturbance value.
        disturb_std (float): The disturbance standard deviation.
        disturb_max (float): The maximum disturbance.
        num_constraints_hard (int): The number of hard constraints.

    Returns:
        tuple: A tuple containing the iteration count, the best combination, the trajectory, and the best reward.
    """

    num_comb = 4
    num_states = len(x_r_list)-1
    num_steps = 2*num_states

    fit_all = []
    comb_all = []
    for i in range(num_comb):
        comb = rand_list_init(num_states)
        comb_new = copy.deepcopy(comb)
        comb_all.append(comb_new)

    fitness_score_table = {}
    traj_temp = {}
    fit_min = 1e6
    for i in range(num_comb):
        comb_appended = copy.deepcopy(comb_all[i])
        comb_appended.append(1)
        traj, _, score, reward, fitness_score_table = fitness_score_lag(comb_appended, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, \
                                                                        beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, 
                                                                        fitness_score_table, mode='genetic')
        fit_all.append(score)
        if traj == {}:
            continue
        elif (traj != {} and score<fit_min):
            fit_min = score
            traj_temp = traj
            comb_min = comb_all[i]
            reward_best = reward
    
    mutation_rate = 0.3
    epsilon = 1e-5
    iteration = 0
    init_run = True
    while init_run == True or traj_temp == {}:
        iteration += 1
        init_run = False
        for i in range(num_steps):
            fit_all = np.array(fit_all)
            p = ((1/(fit_all+epsilon)) / np.sum(1/(fit_all+epsilon))).reshape(-1,)
            comb_all_selct = np.zeros([num_comb,num_states])
            for j in range(0,num_comb):
                comb_all_selct[j,:] = comb_all[np.random.choice(a=np.arange(0,num_comb,1),p=p)]
            new_comb_all = []

            for k in range(0, int(num_comb/2)):
                # Randomly select two combs
                idx_1, idx_2 = np.random.choice(num_comb, size=2, replace=False)
                
                # Perform random exchange between the selected combs
                split = random.randint(0, num_states)
                comb_1 = np.hstack((comb_all_selct[idx_1][0:split], comb_all_selct[idx_2][split:]))
                comb_1 = mutate_process(comb_1, mutation_rate)
                new_comb_1 = copy.deepcopy(comb_1)
                
                comb_2 = np.hstack((comb_all_selct[idx_2][0:split], comb_all_selct[idx_1][split:]))
                comb_2 = mutate_process(comb_2, mutation_rate)
                new_comb_2 = copy.deepcopy(comb_2)
                
                # Append the new exchanged combs into new_comb_all
                new_comb_all.append(new_comb_1)
                new_comb_all.append(new_comb_2)
            
            comb_all = new_comb_all
            fit_all = []
            
            # For each combination of waypoints, calculate its Lagrange Score
            for i in range(num_comb):
                comb_appended = copy.deepcopy(comb_all[i])
                comb_appended.append(1)
                traj, _, score, reward, fitness_score_table = fitness_score_lag(comb_appended, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, \
                                                                        beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, 
                                                                        fitness_score_table, mode='genetic')
                fit_all.append(score)
            if (traj != {} and score<fit_min):
                fit_min = score
                traj_temp = traj
                comb_min = comb_all[i]
                reward_best = reward 
            fit_all = np.array(fit_all)

        comb_min.append(1)
    return iteration, comb_min, traj_temp, reward_best

def deterministic_lag(scenario_num, robot_type, x0, x_r_list, time_horizon, reward_max, radius_list, alpha_values, beta_values, reward_list, U_max, V_max, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):
    """
    Calculates the deterministic lag for a given scenario.

    Parameters:
    - scenario_num (int): The scenario number.
    - robot_type (str): The type of robot.
    - x0 (float): The initial position.
    - x_r_list (list): List of waypoints.
    - time_horizon (float): The time horizon.
    - reward_max (float): The maximum reward.
    - radius_list (list): List of radii.
    - alpha_values (list): List of alpha values.
    - beta_values (list): List of beta values.
    - reward_list (list): List of rewards.
    - U_max (float): The maximum control signal.
    - V_max (float): The maximum speed.
    - obstacle_list (list): List of obstacles.
    - dt (float): The time step.
    - disturbance (bool): Whether to include disturbance.
    - disturb_std (float): The standard deviation of disturbance.
    - disturb_max (float): The maximum disturbance value.
    - num_constraints_hard (int): The number of hard constraints.

    Returns:
    - iteration (int): The number of iterations.
    - best_comb (list): The best combination of constraints.
    - best_traj (dict): The best trajectory.
    - best_reward (float): The best reward.
    """
    
    init_comb = np.ones(shape=(len(x_r_list)))

    fitness_score_table = {}

    best_traj, _, min_r, best_reward, fitness_score_table = fitness_score_lag(init_comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, 
                                                                         beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard,
                                                                        fitness_score_table, mode='deterministic')
    best_comb = init_comb
    dropped_constraints = {}

    # Keep removing the constraint corresponding to the maximum r score until finding a feasible solution
    while best_traj == {}:
        min_r = 1.0e6
        for i in range(len(x_r_list)):
            if i!=len(x_r_list)-1 and dropped_constraints.get(i)==None:
                temp_comb = copy.deepcopy(best_comb)
                temp_comb[i] = 0
                traj, _, r, reward, fitness_score_table = fitness_score_lag(temp_comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, \
                                                                          beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, 
                                                                          fitness_score_table, mode='deterministic')
                if (r <= min_r):
                    if (best_traj!={} and traj!={})or(best_traj=={} and traj=={}) :
                        min_r = r
                        candidate_idx = i
                        best_traj = traj
                        best_reward = reward
                if (traj!={} and best_traj=={}):
                    min_r = r
                    candidate_idx = i
                    best_traj = traj
                    best_reward = reward
        best_comb[candidate_idx] = 0
        dropped_constraints.update({candidate_idx: True})
    iteration = 1
    return iteration, best_comb, best_traj, best_reward

def greedy_lag(scenario_num, robot_type, x0, x_r_list, time_horizon, reward_max, radius_list, alpha_values, beta_values, reward_list, U_max, V_max, \
               obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard):
    """
    Executes the greedy lag algorithm to find the best combination of constraints for a given scenario.

    Args:
        scenario_num (int): The scenario number.
        robot_type (str): The type of robot.
        x0 (float): The initial position of the robot.
        x_r_list (list): List of waypoints.
        time_horizon (float): The time horizon.
        reward_max (float): The maximum reward value.
        radius_list (list): List of radii for each waypoints.
        alpha_values (list): List of alpha values.
        beta_values (list): List of beta values.
        reward_list (list): List of rewards for each waypoints.
        U_max (float): The maximum control signal.
        V_max (float): The maximum speed
        obstacle_list (list): List of obstacles.
        dt (float): The time step.
        disturbance (bool): Flag indicating if disturbance is present.
        disturb_std (float): The standard deviation of the disturbance.
        disturb_max (float): The maximum disturbance value.
        num_constraints_hard (int): The number of hard constraints.

    Returns:
        tuple: A tuple containing the iteration number, the best combination of constraints, the best trajectory, and the best reward.
    """
    init_comb = np.ones(shape=(len(x_r_list)))
    fitness_score_table = {}
    traj, lamda_sum_list, _, best_reward, fitness_score_table = fitness_score_lag(init_comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, 
                                                                         beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard,
                                                                        fitness_score_table, mode='deterministic')
    best_comb = init_comb
    best_traj = traj
    active_constraints = []
    for i in range(len(x_r_list)):
        active_constraints.append(i)   

    # Keep removing constraints corresponding to the largest Lagrange score.
    while traj == {}:
        candidate_idx = np.argmax(np.array([lamda_sum_list]))
        if (active_constraints[candidate_idx] == len(x_r_list)-1):
            lamda_sum_list[candidate_idx] = 0
            candidate_idx = np.argmax(np.array([lamda_sum_list]))
        best_comb[active_constraints[candidate_idx]] = 0
        temp_active_constraints = []
        for idx in active_constraints:
            if idx!=active_constraints[candidate_idx]:
                temp_active_constraints.append(idx)
        active_constraints = temp_active_constraints
        traj, lamda_sum_list, _, best_reward, fitness_score_table = fitness_score_lag(init_comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, 
                                                                         beta_values, reward_list, U_max, V_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard,
                                                                        fitness_score_table, mode='deterministic')
        
    iteration = 1
    best_traj = traj
    return iteration, best_comb, best_traj, best_reward