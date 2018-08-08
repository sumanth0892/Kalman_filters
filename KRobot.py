#Implementing a Kalman Filter to track a robot
#Use a class and functions to implement this
import numpy.random as random
import copy
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot

class KalmanFilter(object):
    #Creating a Kalman Filter with state variables and other noise factors
    def __init__(self,dim_x,dim_z,dim_u = 0):
        self.x = np.zeros((dim_x,1)) #Position vector
        self.P = np.eye(dim_x) #Covariance matrix
        self.Q = np.eye(dim_x) #Process uncertainty
        self.u = np.zeros((dim_x,1)) #Motion vector/Control input
        self.B = 0 #Control transition matrix
        self.F = 0 #State transition matrix
        self.H = 0 #Measurement function
        self.R = np.eye(dim_z) #State uncertainty

        #Identity matrix
        self._I = np.eye(dim_x)

        if use_short_form:
            self.update = self.update_short_form

    def update (self,Z,R=None):
        #Add a new quantitiy Z to the kalman Filter
        if Z is None:
            return
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z)*R

        #Error (Residual measurement)
        y = Z - dot(H,x)

        #System uncertainty into the measurement space
        S = dot(H,P).dot(H.T) + R

        #Map uncertainty into Kalman filter
        K = dot(P,H.T).dot(linalg.inv(S))

        #Predict new x with the residual
        self.x = self.x + dot(K,y)

        I_KH = self._I - dot(K,H)
        self.P = dot(I_KH).dot(P).dot(I_KH.T)

    def predict(self,u=0):
        #Predict the next position
        #u: Optional control vector

        self.x = dot(self.F,self.x) + dot(self.B,u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        



class PosSensor1(object):
    def __init__(self,pos=[0,0],vel=[0,0],noise_scale=1.1):
        self.vel = vel
        self.noise_scale = noise_scale
        self.pos = copy.deepcopy(pos)

    def read(self):
        self.pos[0] = self.pos[0]+self.vel[0]
        self.pos[1] = self.pos[1]+self.vel[1]

        return [self.pos[0]+random.randn() * self.noise_scale,
                self.pos[1]+random.randn() * self.noise_scale]

f1 = KalmanFilter(dim_x=4,dim_z=2)
dt=1.0 #Time step for measurement
f1.F = np.array([[1,dt,0,0],
                 [0,1,0,0],
                 [0,0,1,dt],
                 [0,0,0,1]])
f1.u = 0
f1.H = np.array([[1/0.3048,0,0,0],
                 [0,0,1/0.3048,0]])
f1.R = np.eye(2)*5
f1.Q = np.eye(4)*0.1
f1.x = np.array([[0,0,0,0]]).T
f1.P = np.eye(4)*500

#Initiate the other variables
count = 30
xs,ys=[],[]
pxs,pys=[],[]

s = PosSensor1([0,0],(2,1),1.)

for i in range(count):
    pos = s.read()
    z = np.array([[pos[0]],[pos[1]]])

    f1.predict()
    f1.update()

    xs.append(f1.x[0,0])
    ys.append(f1.x[2,0])
    pxs.append(pos[0]*0.3048)
    pys.append(pos[1]*0.3048)

p1, = plt.plot(xs,ys,'r--')
p2, = plt.plot(pxs,pys)
plt.legend([p1,p2],['filter','measurement'],2)
plt.show()
    
