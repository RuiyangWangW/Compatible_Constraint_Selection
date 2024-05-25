import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter
import sys

sys.path.append("../../src")
from robot_models.SingleIntegrator2D import *
from utils.Safe_Set_Series import *
from predictive_frame_lag import *
from predictive_frame_slack import *
from utils.scenario_disturb import *

t_start = time.perf_counter()
plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.1
t = 0
tf = 50
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.6
alpha_values = [0.4,0.0]
beta_values = [1.8,0.0]
num_constraints_soft1 = 1
robot_type = 'SingleIntegrator2D'
scenario_num = 1
if robot_type == 'SingleIntegrator2D':
    V_max = 0
else:
    V_max = 1.0

# Define Series of Safe Sets
centroids = scenario_waypoints(scenario_num,robot_type)

#Define Obstacles
obstacle_list_x_1 = np.arange(start=-3+0.2,stop=-0.6+0.2, step=0.4)
obstacle_list_y_1 = np.zeros(shape=obstacle_list_x_1.shape)+1.2
obstacle_list_1 = np.vstack((obstacle_list_x_1,obstacle_list_y_1)).T

obstacle_list_y_2 = np.arange(start=-1+0.2, stop=1.4+0.2, step=0.4)
obstacle_list_x_2 = np.zeros(shape=obstacle_list_y_2.shape)+1.2
obstacle_list_2 = np.vstack((obstacle_list_x_2,obstacle_list_y_2)).T
obstacle_list = np.vstack((obstacle_list_1,obstacle_list_2))


num_constraints_hard1 = obstacle_list.shape[0] + 1
radii = np.zeros((centroids.shape[0],))+d_max
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii)

#Define Reward
reward_list = np.array([1,1,1,1,1,1,0])
reward_max = np.sum(reward_list)

#Define Disturbance
disturbance = True
disturb_max = 1.5*U_max
disturb_std = 1.5
f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
f_max_2 = f_max_1/0.5

# Record video if needed
#metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
#writer = FFMpegWriter(fps=15, metadata=metadata)
#movie_name = 'series_of_safesets_small_wind_cbf_online.mp4'


best_idx = 0
best_reward = 0

x0 = np.array([5.0,0.0])

total_reward = 0
total_iter = 0
total_step = 1
for i in range(total_step):
    iteration, best_comb, best_traj, reward = deterministic_lag(robot_type=robot_type, scenario_num=scenario_num, x0=x0, x_r_list=centroids, time_horizon=tf, reward_max=reward_max,radius_list=radii, \
                                                            alpha_values=alpha_values, beta_values=beta_values, reward_list=reward_list, U_max = U_max, V_max = V_max, dt=dt, disturbance=disturbance, \
                                                            disturb_std=disturb_std, disturb_max=disturb_max, obstacle_list=obstacle_list, \
                                                            num_constraints_hard=num_constraints_hard1)

    total_reward += reward
    num_iter = iteration
    total_iter += num_iter
    print('Reward: ', reward )
    print('Time (s): ', time.perf_counter()-t_start)

print('Average Reward: ', total_reward/total_step)
print('Average Iter: ', total_iter/total_step)


# Plot
x_min = -6
x_max = 6
y_min = -2
y_max = 6
fig = plt.figure()
ax = plt.axes(xlim=(x_min,x_max),ylim=(y_min,y_max+2))
ax.set_xlabel("X")
ax.set_ylabel("Y")

rect = patches.Rectangle((-5, y_max), 10, 0.5, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((-3, 1.0), 2.4, 0.4, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((1, -1), 0.4, 2.4, linewidth=1, edgecolor='none', facecolor='k')
ax.add_patch(rect)

for i in range(0,obstacle_list.shape[0]):
    circle = patches.Circle(obstacle_list[i,:], radius=0.2, color='black', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

for i in range(0,centroids.shape[0]):
    if i != centroids.shape[0]-1:
        circle = patches.Circle(centroids[i,:], radius=radii[i], color='blue', zorder=0)
    else:
        circle = patches.Circle(centroids[i,:], radius=radii[i], color='red', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std)/f_max_1 * disturb_max + 4.0
ax.fill_between(x_disturb_1, y_disturb_1, 4.0, alpha=0.2, color='blue')
y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5)/f_max_2 * disturb_max - 0.5
ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')

x_list = best_traj["x"]
y_list = best_traj["y"]
t_list = best_traj["t"]

for i in range(len(best_comb)-1):
    if best_comb[i] == 1:
        centroid = centroids[i,:]
        r = radii[i]
        circle = patches.Circle(centroid, radius=r, color='green', zorder=2)
        ax.add_patch(circle)
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')

ax.axis('equal')
plt.rcParams.update({'font.size': 15})
plt.show()