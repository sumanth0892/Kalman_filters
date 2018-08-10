#Building a simulator to track a ball through air
#Ignoring air friction
from math import radians,sin,cos
import math
from numpy import random
import matplotlib.pyplot as plt
import copy
import scipy.linalg as linalg
import numpy as np
from numpy import dot

def runge_kutta(y,x,dx,f):
    #Computes the fourth order Runge-Kutta
    #y = y0
    #x = x0
    #dx is the difference in time step
    #f: Function computing dy/dx
    k1 = dx*f(y,x)
    k2 = dx*f(y+0.5*k1,x+0.5*dx)
    k3 = dx*f(y+0.5*k2,x+0.5*dx)
    k4 = dx*f(y+k3,x+dx)

    return y+(k1+2*k2+2*k3+k4)/6

def fx(x,t):
    return fx.vel
def fy(y,t):
    return fy.vel - 9.8*t

class PosSensor1(object):
    def __init__(self,pos=[0,0],vel=[0,0],noise_scale=1.1):
        self.vel = vel
        self.noise_scale = noise_scale
        self.pos = copy.deepcopy(pos)

    def read(self):
        self.pos[0] = self.pos[0]+self.vel[0]
        self.pos[1] = self.pos[1]+self.vel[1]
        return[self.pos[0]+random.randn()*self.noise_scale,
               self.pos[1]+random.randn()*self.noise_scale]



class KalmanFilter(object):
    #Creating a Kalman filter with state variables
    def __init__(self,dim_x,dim_z,dim_u=0):
        self.x = np.zeros((dim_x,1)) #Position vector
        self.P = np.eye(dim_x) #Covariance matrix
        self.Q = np.eye(dim_x) #Process uncertainty
        self.u = np.zeros((dim_x,1)) #Control vector
        self.B = 0 #Transition matrix
        self.F = 0 #State transition matrix
        self.H = 0 #measurement function
        self.R = np.eye(dim_z) #state uncertainty

        #Identity matrix
        self._I = np.eye(dim_x)

    def update(self,Z,R=None):
        #Add a new quanitity Z to the measurement
        if Z is None:
            return
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_Z)*R

        #Error measurement/Residual
        y = Z = dot(self.H,self.x)

        #System uncertainty into measurement space
        S = dot(self.H,self.P).dot(self.H>T) + R

        #Map uncertainty onto Kalman space and calculate Kalman gain
        K = dot(self.P,self.H.T).dot(linalg.inv(S))

        #Update the value of x
        self.x = self.x + dot(K,y)

        I_KH = self._I - dot(K,self.H)
        self.P = (I_KH).P.I_KH.T + dot(K,R).dot(K.T)

    def predict(self,u=0):
        #Predict the next position
        #u: Optional control input vector
        self.x = dot(self.F,self.x) + dot(self.B,u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        

class BallTrajectory2D(object):
    def __init__(self,x0,y0,velocity,theta_deg=0,g=9.8,noise=[0.0,0.0]):
        self.x = x0
        self.y = y0
        self.t = 0
        theta = math.radians(theta_deg)
        fx.vel = math.cos(theta)*velocity
        fy.vel = math.sin(theta)*velocity
        self.g = g
        self.noise = noise

    def step(self,dt):
        self.x = runge_kutta(self.x,self.t,dt,fx)
        self.y = runge_kutta(self.y,self.t,dt,fy)
        self.t+=dt
        return(self.x+random.randn()*self.noise[0],self.x+random.randn()*self.noise[1])
    

#Create a trajectory of a ball starting at [0,15] with a velocity of 60m/s at an angle of 65deg

    
def ball_kf(x,y,omega,v0,dt,r=0.5,q=0):
    g = 9.8 #Gravitational constant
    f1 = KalmanFilter(dim_x=5,dim_z=2)

    ay = 0.5*dt**2
    f1.F = np.mat([[1,dt,0,0,0],
                   [0,1,0,0,0]
                   [0,0,1,dt,ay],
                   [0,0,0,1,dt],
                   [0,0,0,0,1]])

    f1.H = np.mat([[1,0,0,0,0],
                   [0,0,1,0,0]])
    f1.R*=r
    f1.Q*=q

    omega = radians(omega)
    vx = np.cos(omega)*v0
    vy = np.sin(omega)*v0

    f1.x = np.mat([x,vx,y,vy,-9.8]).T

    return f1

y=1
x=0
theta = 35
v0 = 80
dt=0.1
ball = BallTrajectory2D(x0=x,y0=y,theta_deg = theta,velocity=v0,noise=[0.2,0.2])
f1 = ball_kf(x,y,theta,v0,dt)
t=0
xs=[]
ys=[]

while f1.x[2,0] > 0:
    t+=dt
    x,y = ball.step(dt)
    z = np.mat([[x,y]]).T

    f1.update(z)

    xs.append(f1.x[0,0])
    ys.append(f1.x[2,0])

    f1.predict()
    p1 = plt.scatter(x,y,color='green',marker='+',s=0.75,alpha = 0.5)
p2, = plt.plot(xs,ys,lw=2)
plt.legend([p2,p1],['Kalman Filter','Measurements'])
plt.show()
