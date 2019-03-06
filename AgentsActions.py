#Agents.py
#An agent observes the world and carries out actions in the environment
#The agent internally updates as well.
#The environment takes in actions and updates as well.
import random
class Agent:
    def __init__(self,env):
        self.env = env
        #Set up the agent

    def go(self,n):
        #Acts for n time steps
        raise NotImplementedError("go") #Abstract


from display import Displayable
class Environment(Displayable):
    def initial_percepts(self):
        raise NotImplementedError("Initial_percepts")

    def do(self,action):
        raise NotImplementedError("do")

class TP_env(Environment):
    prices = []
    max_price_addon = 20

    def __init__(self):
        #Paper buying agent
        self.time = 0
        self.stock = 20
        self.stock_history = []
        self.price_history = []

    def initial_percepts(self):
        #Return the initial perceptions
        self.stock_history.append(self.stock)
        price = self.prices[0] + random.randrange(self.max_price_addon)
        self.price_history.append(price)
        return {'price':price,
                'instock':self.stock}

    def do(self,action):
        #Buys stocks and returns percepts
        used = pick_from_dist({6:0.1,5:0.1,4:0.2,3:0.3,2:0.2,1:0.1})
        bought = action['buy']
        self.stock = self.stock+bought-used
        self.stock_history.append(self.stock)
        self.time+=1
        price = (self.prices[self.time%len(self.prices)]#repeating patterns
                 +random.randrange(self.max_price_addon)
                 +self.time//2)
        self.price_history.append(price)
        return {'price':price,'instock':self.stock}

def pick_from_dist(item_prob_dist):
    #Returns a value from distribution
    ranreal = random.random()
    for (it,prob) in item_prob_dist.items():
        if ranreal<prob:
            return it
        else:
            ranreal-=prob
    raise RuntimeError(str(item_prob_dist)+"is not a probability distribution")

class TP_agent(Agent):
    #The agent doesn't have access to price models but observes the prices
    def __init__(self,env):
        self.env = env
        self.spent = 0
        percepts = env.initial_percepts()
        self.ave = self.last_price = percepts['price']
        self.instock = percepts['instock']

    def go(self,n):
        #Go for n time steps
        for i in range(n):
            if self.last_price<0.9*self.ave and self.instock<60:
                tobuy = 48
            elif self.instock<12:
                tobuy = 12
            else:
                tobuy = 0
            self.spent+= tobuy*self.last_price
            percepts = env.do({'buy':tobuy})
            self.last_price = percepts['price']
            self.ave = self.ave+(self.last_price-self.ave)*0.05
            self.instock = percepts['instock']

env = TP_env()
ag = TP_agent(env)

        
