#A Kalman Filter using State-spaces
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class KalmanFilter(object):
    def __init__(self,dim_x,dim_z,dim_u=0):
        #dim_x: Number of state variables measured
        #dim_x: Number of measurement inputs
        #dim_u: Number of control inputs
        self.x = np.zeros((dim_x,1)) #State vector
        self.P = np.eye(dim_x) #Uncertainty Covariance
        self.Q = np.eye(dim_x) #Process Noise
        self.u = np.zeros((dim_x,1)) #Control Input
        self.B = 0 #Control transition matrix
        self.H = 0 #Measurement Function
        self.R = np.eye(dim_z) #State uncertainty/Noise

        #Identity matrix
        self._I = np.eye(dim_x)


    def update(self,Z,R=None):
        #This is the update step of the Kalman Filter
        if Z is None:
            return
        
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z)*R

        #Error measurement
        y = Z - np.dot(self.H,self,x)

        #System uncertainty into measurement space
        S = np.dot(np.dot(self.H,self.P),(self.H).T)

        #Map uncertainty into Kalman Gain
        K = np.dot(np.dot(P,H.T),np.linalg.inv(S))

        #Predict new x with the residual
        self.x = self.x + np.dot(K,y)

        I_KH = self._I - np.dot(K,H)
        self.P = np.dot(I_KH,self.P)

    def predict(self,u=0):
        self.x = np.dot(self.F,self.x)
        self.P = np.dot(np.dot(F,P),F.T) + self.Q
        
        
