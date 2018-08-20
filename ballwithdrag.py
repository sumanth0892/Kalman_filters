from math import math,exp,cos,sin,radians

def mph_to_mps(x):
    return x*0.447

def drag_force(velocity):
    #Returns the force on a baseball due to air drag at a particular velocity
    return (0.0039+0.0058/(1.0 + exp((velocity-35.0)/5.0))) * velocity

v = mph_to_mps(110.0)
y=1
x=0
dt=0.1
theta = radians(35)

def solve(x,y,vel,v_wind,launch_abgle):
    xs=[]
    ys=[]
    v_x = vel*cos(launch_angle)
    v_y  = vel*sin(launch_angle)

    while (y>=0):
        #Euler equations
        x += v_x*dt
        y += v_y*dt

        #Force due to air drag
        velocity = sqrt((v_x - v_wind)**2 + v_y**2)
        F = drag_force(velocity)

        #Euler equations for vx and vy
        v_x = v_x + F*(v_x - v_wind)*dt
        v_y = v_y - 9.8*dt - F*v_y*dt
        xs.append(x)
        ys.append(y)
    return xs,ys

x,y = solve(x=0,y=1,vel=v,v_wind=0,launch_angle=theta)
plt.scatter(x,y)
