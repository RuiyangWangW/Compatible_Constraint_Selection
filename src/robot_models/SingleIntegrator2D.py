import numpy as np

class SingleIntegrator2D:
    
    def __init__(self,X0,dt,ax,id,num_constraints_hard = 1, num_constraints_soft = 1, color='r',palpha=1.0,plot=True):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        self.nextU = self.U

        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=10,zorder=10)
            self.render_plot()
        
        
        self.A1_hard = np.zeros((num_constraints_hard,2))
        self.b1_hard = np.zeros((num_constraints_hard,1))
        self.A1_soft = np.zeros((num_constraints_soft,2))
        self.b1_soft = np.zeros((num_constraints_soft,1))
        self.slack_constraint = np.zeros((num_constraints_soft,1))
        
        self.Xs = X0.reshape(-1,1)
        self.Us = np.array([0,0]).reshape(-1,1)        
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        if self.plot == True:
            self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])

            # scatter plot update
            self.body.set_offsets([x[0],x[1]])

    def lyapunov(self, G):
        V = -np.linalg.norm( self.X - G[0:2] )**2
        dV_dx = -2*( self.X - G[0:2] ).T
        return V, dV_dx
    
    def nominal_input(self,G):
        V, dV_dx = self.lyapunov(G)
        return 5.0 * dV_dx.reshape(-1,1)
    
    def static_safe_set(self, target, d_max):
        h =  d_max**2 - np.linalg.norm(self.X[0:2] - target[0:2])**2
        dh_dx = -2*( self.X - target[0:2] ).T

        return h , dh_dx