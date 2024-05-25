import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cvxpy as cp

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
    
class DoubleIntegrator2D: 
    def __init__(self,X0,dt,ax,V_max,num_constraints_soft,num_constraints_hard,color='r',palpha=1.0,plot=False): 

        ''' X0: iniytial state dt: simulation time step ax: plot axis handle ''' 

        self.X = X0.reshape(-1,1) 
        self.dt = dt 
        self.type = 'DoubleIntegrator2D'
        self.ax = ax
        self.V_max = V_max 
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=10,zorder=10)
            self.render_plot()
        
        self.A1_hard = np.zeros((num_constraints_hard,2))
        self.b1_hard = np.zeros((num_constraints_hard,1))
        self.A1_soft = np.zeros((num_constraints_soft,2))
        self.b1_soft = np.zeros((num_constraints_soft,1))


    def render_plot(self): 
        x = np.array([self.X[0,0],self.X[1,0]]) 
        self.body.set_offsets([x[0],x[1]]) 
    
    def f(self): 
        dx12_dt = np.array([self.X[2]*np.cos(self.X[3]), self.X[2]*np.sin(self.X[3])])
        return np.vstack((dx12_dt,np.zeros((2,1)))).reshape(-1,1) 
    
    def J(self): 
        return np.array([[np.cos(self.X[3]), -self.X[2]*np.sin(self.X[3])], 
                                  [np.sin(self.X[3]), self.X[2]*np.cos(self.X[3])]], dtype='float64').reshape(-1,2)
            
    # Move the robot  
    def step(self, U, U_d, disturbance): 
        if disturbance:
            x_ddot = self.J()@U + U_d
            U_eff = np.linalg.pinv(self.J())@x_ddot
            self.X = self.X + (self.f() + np.vstack([np.zeros((2,1)), U_eff])) * self.dt
        else:
            self.X = self.X + (self.f() + np.vstack([np.zeros((2,1)), U])) * self.dt

        if self.X[2] > self.V_max:
            self.X[2] = self.V_max
        elif self.X[2] < -self.V_max:
            self.X[2] = -self.V_max

        if self.plot == True:
            self.render_plot()
        return self.X
    
    def lyapunov(self, G):
        phi_0 = -np.linalg.norm(self.X[0:2]-G[0:2])**2
        dphi_0_dx = -2*(self.X[0:2]-G[0:2])
        dx12_dt = np.array([self.X[2]*np.cos(self.X[3]), self.X[2]*np.sin(self.X[3])]).reshape(-1,1)
        return phi_0, dphi_0_dx, dx12_dt
    
    def barrier(self, obs, d_min):
        h = np.linalg.norm(self.X[0:2] - obs[0:2])**2 - d_min**2 
        dh_dx = 2*(self.X[0:2]-obs[0:2])
        dx12_dt = np.array([self.X[2]*np.cos(self.X[3]), self.X[2]*np.sin(self.X[3])]).reshape(-1,1)
        return h, dh_dx, dx12_dt
    
    def nominal_input(self, x_r):
        k_omega = 3.0
        k_x = 0.5
        k_v = 1.5
        distance = max(np.linalg.norm(self.X[0:2]-x_r[0:2]), 0.1)
        desired_heading = np.arctan2(x_r[1]-self.X[1], x_r[0]-self.X[0])
        error_heading = wrap_angle(desired_heading - self.X[3])
        omega = k_omega*error_heading*np.tanh(distance)
        speed = k_x*distance*np.cos(error_heading)
        u_r = 1.0*k_v*(speed-self.X[2])
        return np.array([u_r, omega]).reshape(-1,1)
        