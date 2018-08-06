import numpy as np
import matplotlib.pyplot as plt

def predict(a,b,c,d):
    return (a+c,b+d)
def update(mu1,var1,mu2,var2):
    mu=(mu1*var2 + mu2*var1)/(var1+var2)
    var = 1/((1/var1)+(1/var2))
    return mu,var

class PetSensor(object):
    def __init__(self,pos,speed,noise):
        self.pos = pos
        self.speed = speed
        self.noise = np.sqrt(noise)
    def sense(self):
        self.pos = self.pos+self.speed
        return self.pos+np.random.randn()*self.noise

pos=(0,500) #initial estimate and Variance (error)
movement = 1
movement_error=2
sensor_error=10
pet = PetSensor(pos[0],movement,sensor_error)
ps=[]
zs=[]
vs=[]

for i in range(100):
    pos=predict(pos[0],pos[1],movement,movement_error)
    

    Z = pet.sense()
    zs.append(Z)
    vs.append(pos[1])

    pos = update(pos[0],pos[1],Z,sensor_error)
    ps.append(pos[0])

plt.plot(zs,'blue',label='Measured data')
plt.plot(ps,'red',label='Predicted positions')
plt.plot(vs,'green',label='Variance')
plt.legend(loc='best')
plt.title('Measured vs Predicted positions')
plt.grid(True,color='black')
plt.show()
