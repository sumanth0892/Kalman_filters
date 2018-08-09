#Building a simulator to track a ball through air
#Ignoring air trajectory
from math import radians,sin,cos
import math
from numpy import random
import matplotlib.pyplot as plt

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
x=0
y=15
traj = BallTrajectory2D(x0=x,y0=y,theta_deg=65,velocity=100,noise=[0.0,0.0])
t=0
dt=0.25
while y>=0:
    x,y = traj.step(dt)
    t+=dt
    if y>=0:
        plt.scatter(x,y)

plt.axis('equal')
plt.show()

    
