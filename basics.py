import numpy as np
import matplotlib.pyplot as plt
class PetSensor(object):
    def __init__(self,pos0,speed,noise):
        self.pos = pos0
        self.speed = speed
        self.noise = np.sqrt(noise)
    def sense(self):
        self.pos = self.pos+self.speed
        return self.pos+np.random.randn()*noise

pet = PetSensor(0,1,0.5)
noise = 0.50000
xs=[]
print ("Sensor data indicates:", "\n")
for i in range(100):
    position = pet.sense()
    xs.append(position)
    print("%0.4f" %xs[i],)

plt.plot([0,99],[1,100],'r--',label='Actual path')
plt.plot(xs,'blue',label='Sensor data')
plt.legend(loc='best')
plt.grid(True,color='black')
plt.title("Noise is %0.4f" %noise)
plt.show()

        
