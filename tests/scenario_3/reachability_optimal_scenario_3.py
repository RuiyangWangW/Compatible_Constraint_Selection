import numpy as np
import math
import time
import copy
import multiprocessing
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from matplotlib.animation import FFMpegWriter
from discretize_helper_scenario_3 import *
import sys
sys.path.append("../../src")

from robot_models.SingleIntegrator2D import *
from utils.Safe_Set_Series import *
from utils.scenario_disturb import *

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.2
tf = 60
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.4
scenario_num = 1
robot_type = 'SingleIntegrator2D'

# Plot                  
plt.ion()
x_min = -6
x_max = 6
y_min = -2
y_max = 6
fig = plt.figure()
ax = plt.axes(xlim=(x_min,x_max),ylim=(y_min,y_max+2)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Define Series of Safe Sets
centroids = scenario_waypoints(scenario_num,robot_type)

rect = patches.Rectangle((-5, y_max), 10, 0.5, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((-3, 1.0), 2.4, 0.4, linewidth=1, edgecolor='none', facecolor='k')
ax.add_patch(rect)
rect = patches.Rectangle((1, -1), 0.4, 2.4, linewidth=1, edgecolor='none', facecolor='k')
ax.add_patch(rect)

centroids_comb = []
radii_comb = []
alpha_comb = []

comb = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
for i in range(len(comb)):
    if comb[i] == 1:
        centroids_comb.append(centroids[i,:])
        radii_comb.append(d_max)
        alpha_comb.append(1.0)

centroids_comb = np.array(centroids_comb)
radii_comb = np.array(radii_comb)
alpha_comb = np.array(alpha_comb)
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids_comb,radii=radii_comb,alpha_list=radii_comb)
num_active_points = len(radii_comb)

for i in range(0,centroids_comb.shape[0]):
    if i != centroids_comb.shape[0]-1:
        circle = patches.Circle(centroids_comb[i,:], radius=d_max, color='blue', zorder=0)
    else:
        circle = patches.Circle(centroids_comb[i,:], radius=d_max, color='red', zorder=10)
    ax.add_patch(circle)
circle = patches.Circle(centroids[4,:], radius=d_max, color='blue', zorder=0)
ax.add_patch(circle)
circle = patches.Circle(centroids[5,:], radius=d_max, color='blue', zorder=0)
ax.add_patch(circle)
circle = patches.Circle(centroids[7,:], radius=d_max, color='blue', zorder=0)
ax.add_patch(circle)

#Define Disturbance
disturbance = True
disturb_max = 0.9*U_max
disturb_std = 1.5
f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
f_max_2 = f_max_1/0.5

x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std)/f_max_1 * disturb_max + 4.0
ax.fill_between(x_disturb_1, y_disturb_1, 4.0, alpha=0.2, color='blue')

y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5)/f_max_2 * disturb_max - 0.6
ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')

# If recording an video
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_with_large_wind.mp4'

#Define Search Map
control_hash_table = {}
in_control_hash_table = {}
step = 0.1
x_range = np.arange(start=x_min, stop=x_max+step, step=step)
y_range = np.arange(start=y_min, stop=y_max+step, step=step)
feasible_candidates = []

# Define feasible candidates in x and y range
for x in x_range:
    for y in y_range:
        if x > -3.0 and x < -0.6 and y < 1.4 and y > 1.0:
            continue
        if x > 1.0 and x < 2.0 and y < 1.4 and y > -1.0:
            continue
        x0 = np.array([x,y])
        feasible_candidates.append(x0)

# For each of the feasible candidates, finding its forward set
with multiprocessing.Pool() as pool:
    for (x0_key, forward_set, dist_ford) in pool.map(discretize_u_forward_cal,feasible_candidates):
        # For each forward cell of a feasible candidate, check if it is within the bounds of the search map
        # add x0 to the backward set of the forward cell if the forward cell is within the bounds
        for idx, forward_cell in enumerate(forward_set):
            x = ""
            for i in range(len(forward_cell)):
                a = forward_cell[i]
                if a!=',':
                    x += a
                else:
                    break
            y = forward_cell[i+1:]
            x = x_range[int(x)]
            y = y_range[int(y)]
            if y > y_max or y < y_min or x > x_max or x < x_min:
                continue
            if x > -3.0 and x < -0.6 and y < 1.4 and y > 1.0:
                continue
            if x > 1.0 and x < 2.0 and y < 1.4 and y > -1.0:
                continue
            if (in_control_hash_table.get(forward_cell)==None):
                backward_set = np.array([x0_key])
                dist_back = np.array([dist_ford[idx]])
            else:
                backward_set, dist_back = control_hash_table.get(forward_cell)
                backward_set = np.append(backward_set,np.array([x0_key]))
                dist_back = np.append(dist_back,np.array([dist_ford[idx]]))
            control_hash_table.update({forward_cell: (backward_set, dist_back)})
            in_control_hash_table.update({forward_cell: True})

x0 = np.array([5.0,0.0])
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=False)

final_centroids = Safe_Set_Series.centroids[-1,:]
final_target_centroid = np.array([final_centroids]).reshape(2,1)
r = Safe_Set_Series.radii[-1]
x_final_target_range = np.arange(start=final_target_centroid[0]-r,stop=final_target_centroid[0]+r+step,step=step)
y_final_target_range = np.arange(start=final_target_centroid[1]-r,stop=final_target_centroid[1]+r+step,step=step)

# Obtain all possible x-y coordinates that are within the radius of the final target
success_list = np.array([])
pos_in_success_table = {}
for x in x_final_target_range:
    for y in y_final_target_range:
        if ((x-final_target_centroid[0])**2 + (y-final_target_centroid[1])**2) <= r**2:
            target_pos = np.array([x,y]).reshape(2,1)
            target_pos_key = str(int((target_pos[0]-x_min)/step))+","+str(int((target_pos[1]-y_min)/step))
            success_list = np.append(success_list,np.array([target_pos_key]))
            pos_in_success_table.update({target_pos_key: True})

# Backward Reachability Search to obtain all possible x-y coordinates that can reach any x-y coordinates within the radius of the final target
while success_list.size > 0:
    current = success_list[0]
    success_list = np.delete(success_list, obj=0, axis=0)
    print(success_list.size)
    if (in_control_hash_table.get(current)==None):
        continue
    else:
        backward_set, _ = control_hash_table.get(current)
    filtered_backward_set = None
    for i in range(backward_set.size):
        has_been_pushed = pos_in_success_table.get(backward_set[i])
        if has_been_pushed==None:
            none_list = np.array([filtered_backward_set == None]).reshape(-1,).tolist()
            if any(none_list):
                filtered_backward_set = np.array([backward_set[i]])
            else:
                filtered_backward_set = np.append(filtered_backward_set,np.array([backward_set[i]]),axis=0)                
            pos_in_success_table.update({backward_set[i]: True})

    none_list = np.array([filtered_backward_set == None]).reshape(-1,).tolist()
    if any(none_list):
        continue
    if len(success_list)> 0:
        success_list = np.append(success_list,filtered_backward_set,axis=0)
    else:
        success_list = filtered_backward_set

x_success_list = []
y_success_list = []
for i, pos in enumerate(pos_in_success_table):
    current = pos
    x = ""
    for i in range(len(current)):
        a = current[i]
        if a!=',':
            x += a
        else:
            break
    y = current[i+1:]
    x = x_range[int(x)] 
    y = y_range[int(y)]
    x_success_list.append(x)
    y_success_list.append(y)

reward = 0
active_safe_set_id = 0
final_path = []
chosen_node = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))

delta_t_limit = float(tf)/len(radii_comb)
delta_t = 0
max_iter = 1e5
x_list = []
y_list = []
t_list = []
t = 0


# Generate path using Dijkstra's algorithm for the robot to move from its current position to the assigned waypoint 
# with a check on that it can reach the final waypoint after arriving at the current waypoint
with writer.saving(fig, movie_name, 100): 
    for i in range(num_steps):

        current_pos = robot.X
        current_pos_key = chosen_node

        if Safe_Set_Series.id != active_safe_set_id:
            Safe_Set_Series.id = active_safe_set_id
            centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
            r = Safe_Set_Series.return_radius(Safe_Set_Series.id)
            circle = patches.Circle(centroid, radius=r, color='blue', zorder=1)
            ax.add_patch(circle)
        
        centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
        r = Safe_Set_Series.return_radius(Safe_Set_Series.id)
        
        if len(final_path) == 0:
            possible_node_list = PriorityQueue()
            in_path_list = {}
            x_cent_range = np.arange(start=centroid[0]-r,stop=centroid[0]+r+step,step=step)
            y_cent_range = np.arange(start=centroid[1]-r,stop=centroid[1]+r+step,step=step)
            for x in x_cent_range:
                for y in y_cent_range:
                    if ((x-centroid[0])**2+(y-centroid[1])**2) <= r**2:
                        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
                        in_success_table = pos_in_success_table.get(pos_key)
                        in_control_table = in_control_hash_table.get(pos_key)
                        if (in_success_table) and in_control_table:
                            possible_node_list.put((0, [pos_key]))
                            in_path_list.update({pos_key: True})

            if possible_node_list.empty():
                active_safe_set_id += 1
                continue
            iter_i = 0
            while ~possible_node_list.empty():
                iter_i += 1
                prev_weight, possible_path = possible_node_list.get()
                node = possible_path[-1]

                if node == current_pos_key:
                    final_path = possible_path
                    final_path.pop(-1)
                    break
                
                if iter_i >= max_iter:
                    final_path = []
                    print('too long')
                    break

                if (in_control_hash_table.get(node)==None):
                    continue
                else:
                    backward_set,dist_back = control_hash_table.get(node)

                for idx,cell in enumerate(backward_set):
                    new_path = copy.deepcopy(possible_path)
                    new_path.append(cell)
                    weight = dist_back[idx] + prev_weight
                    if in_path_list.get(cell) != None:
                        curr_weight = in_path_list.get(cell)
                        if (weight < curr_weight):
                            possible_node_list.put((weight, new_path))
                        else:
                            continue
                    else:
                        possible_node_list.put((weight, new_path))
                    in_path_list.update({cell: weight})

        if final_path == []:
            if active_safe_set_id < num_active_points-1:
                active_safe_set_id += 1
            else:
                break
            continue
        chosen_node = final_path.pop(-1)
        chosen_x = ""
        for i in range(len(chosen_node)):
            a = chosen_node[i]
            if a!=',':
                chosen_x += a
            else:
                break
        chosen_y = chosen_node[i+1:]
        chosen_x = x_range[int(chosen_x)] 
        chosen_y = y_range[int(chosen_y)]
        robot.X = np.array([chosen_x,chosen_y]).reshape(-1,1)
        x_list.append(chosen_x)
        y_list.append(chosen_y)
        #robot.render_plot()
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        #print(active_safe_set_id)
        delta_t += dt
        t += dt
        t_list.append(t)


        if (delta_t>delta_t_limit):
            if active_safe_set_id < num_active_points-1:
                active_safe_set_id += 1
                final_path = []
                delta_t = 0
                continue
            else:
                break
            

        if len(final_path)==0:
            centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
            r = Safe_Set_Series.return_radius(Safe_Set_Series.id)
            circle = patches.Circle(centroid, radius=r, color='green', zorder=2)
            ax.add_patch(circle)
            reward += 1
            print("ID: ", Safe_Set_Series.id)
            delta_t = 0
            if active_safe_set_id < num_active_points-1:
                active_safe_set_id += 1
            else:
                break
                
        
        writer.grab_frame()


print(len(pos_in_success_table))
print(len(x_success_list))

print('reward', reward-1)
ax.axis('equal')
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')
plt.ioff()
plt.show()

