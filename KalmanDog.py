#Designing a Boilerplate Kalman Filter from scratch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self,dim_x,dim_z,R):
        self.x = np.zeros((dim_x,1))
        self.F = np.eye(dim_x)
        self.H =
        self.P =
        self.u =
        self.B =
        self.R =
        self.Q =

        self._I = np.eye(dim_x)

        if use_short_form:
            self.update = self.use_short_form

    def update(self,Z,R=None):
        #Add a new measurement 'Z' which is a measure from the sensor
        y = Z - np.dot(H,x)
        S = dot(H,P).dot(H.T) + R

        #Compute Kalman Gain
        K = np.dot(np.dot(P,H.T),np.linalg.inv(S))

        #Update the value of x
        self.x = self.x + K*y
        self.P = np.dot(self._I-np.dot(K,H))

    def predict(self):
        self.x =
        self.P =
        return self.x,self.P

def main():
    R=[]
    pos = np.array(a,b)
    ps=[]
    vs=[]
    var_s=[]
    varp=[]
    dim_x=pos.ndim

    for i in range(100):
        Z = measure()
        move = KalmanFilter(dim_x,Z.ndim,R)
        P,C = move.predict()
        ps.append(P[0])
        vs.append(P[1])
        var_s.append(C[0,0])
        varp.append(C[1,1])

plt.plot(ps,color=' ')
plt.plot(vs,color=' ')
plt.plot(var_s,color=' ')
plt.plot(varp,color=' ')

        
