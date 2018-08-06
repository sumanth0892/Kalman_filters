import numpy as np
import matplotlib.pyplot as plt

def update(mu1,var1,mu2,var2):
    mu = (mu1*var2 + mu2*var1)/(var1+var2)
    var = 1/((1/var1)+(1/var2))
    return mu,var

def predict(a,b,c,d):
    return (a+c,b+d)

class TempSensor(object):
    def __init__(self,ti,change,noise):
        self.temp = ti
        self.change = change
        self.noise = np.sqrt(noise)

    def sense(self):
        self.temp = self.temp+self.change
        return self.temp+self.change*np.random.randn()

T=[25,1000] #Initial estimate of the temperature
change=0
noise = 2.13**2
movement_error=0.2
temp = TempSensor(T[0],change,noise)
Ts=[]
Zs=[]
Vs=[]
N=50

for i in range(N):
    Z = temp.sense()
    Zs.append(Z)

    T = update(T[0],T[1],Z,noise)
    Ts.append(T[0])
    Vs.append(T[1])
    T = predict(T[0],T[1],change,movement_error)

plt.scatter(range(N),Zs,marker='+',label='Measurement')
plt.plot(Ts,'green',label='Filter')
plt.title('Temperature values')
plt.legend(loc='best')
plt.grid(True,color='black')
plt.show()
plt.plot(Vs)
plt.show()

