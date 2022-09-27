import random as rd
import math as m
import time as t
import numpy as np
import matplotlib.pyplot as plt

#Define objective function f

def f(x):
    result = 0
    if function=='B' or function=='b':
        result = (x[0] + 2*x[1] -7)**2 + (2*x[0] + x[1] - 5)**2  #Booth
    elif function=='A' or function == 'a':
        result = -20 * m.exp(-0.2*m.sqrt(0.5*(x[0]**2+x[1]**2))) - m.exp(0.5*(m.cos(2*m.pi*x[0]) + m.cos(2*m.pi*x[1]))) + m.e + 20 #Ackley
    elif function == 'S' or function == 's':
        for i in range(dimensions):
            result = result + x[i]**2   #Sphere
    elif function == 'R' or function == 'r':
        for i in range(dimensions):
            result = result + (x[i]**2 - 10*m.cos(2*m.pi*x[i]))     #Rastrigin
        result = result + 10*dimensions
    elif function == 'E' or function == 'e':
        result = -m.cos(x[0])*m.cos(x[1])*m.exp(-(x[0]-m.pi)**2 - (x[1]-m.pi)**2)   #Easom
    return result


#Particle Class

class Particle():

    #Constructor that initializes particle's position and velocity
    #position and velocity parameters are initital particle's position and velocity
    def __init__(self,position,velocity):
        self.position = position[:]                 #Particle's position
        self.velocity = velocity[:]                 #Particle's velocity
        self.best = self.position[:]                #Best self position
        self.positionFitness = f(self.position)     #Current position fitness
        self.bestFitness = f(self.best)             #Best self position fitness

    #Check if current position is personal best
    def Best(self):
        if self.positionFitness < self.bestFitness:
            self.best = self.position[:]
            self.bestFitness = self.positionFitness       
        
    #Update particle's position
    def setPosition(self):
        for i in range(dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
            
    #Calculate particle's Fitness value.
    def evaluate(self):
        self.positionFitness = f(self.position)

    #Set particles position, if it is out of boundaries, according to absorbing wall technique
    def fixPosition(self):
        for i in range(dimensions):
            if self.position[i] < L[i]:
                self.position[i] = L[i]
            elif self.position[i]> U[i]:
                self.position[i] = U[i]


    #Reset particle's velocity if it is out of boundaries
    def fixVelocity(self):
        for i in range(dimensions):
            if self.velocity[i] > vmax[i]:
                self.velocity[i] = vmax[i]
            elif self.velocity[i] < -vmax[i]:
                self.velocity[i] = -vmax[i]
        

#Predator Class. This is subclass of particles class

class Predator(Particle):

    #Constructor that calls base's class constructor.
    def __init__(self,position,velocity):
        super().__init__(position,velocity)

    #Calculate Predator's Velocity according to formula. preyBest Parameter is the best position of Preys swarm
    def setVelocity(self,preyBest):
        r = [rd.uniform(0,vmax[i]) for i in range(dimensions)]           #Random variable used to update predators velocity
        for i in range(dimensions):
            self.velocity[i] = r[i]* (preyBest[i] - self.position[i])  
        

#Prey Class. This is subclass of particles class
    
class Prey(Particle):

    #Constructor that calls base's class constructor and initialize variables used in the prey's velocity formula
    def __init__(self,position,velocity):
        super().__init__(position,velocity)
        self.c1 = 2                 #Cognitive Coefficient
        self.c2 = 2                 #Social Coefficient
        self.c3 = 2.5               #Prey Coefficient
       
    #Find the nearest particle element from predators list to self prey.
    def EuclideanDistance(self,predators):
        for i in range(len(predators)):
            distance = 0
            for j in range(dimensions):
                distance = distance + (self.position[j] - predators[i].position[j])**2          
            distance = m.sqrt(distance)
            if i == 0:
                d  = distance
            elif distance<d:
                d = distance
        return d
    
    #Calculate the measure of the effect that the predator has on the prey. d parameter is distance from nearest predator. a and b parameters are distance coefficients.
    def D(self,d,a,b):
        return a*m.exp(-b*d)

    #Calculate Preys's Velocity according to formula. pBest parameter is prey's swarm best position and w parameter is inertia rate. d, a and b are  aparameters used for function D.
    #pf parameter is fear propability
    def setVelocity(self,pBest,d,a,b,w,pf):
        rn = rd.uniform(0,1)
        r1 = [rd.uniform(0,1) for i in range(dimensions)]    
        r2 = [rd.uniform(0,1) for i in range(dimensions)]        #r1, r2, r3 and rn are random variables 
        r3 = [rd.uniform(0,1) for i in range(dimensions)]        #used to update preys velocity

        if rn <= pf:
            for i in range(dimensions):
                self.velocity[i] =  w*self.velocity[i] + self.c1*r1[i]*(self.best[i] - self.position[i]) + self.c2*r2[i]*(pBest[i] - self.position[i]) + self.c3*r3[i]*self.D(d,a,b)
        else:
             for i in range(dimensions):
                self.velocity[i] =  w*self.velocity[i] + self.c1*r1[i]*(self.best[i] - self.position[i]) + self.c2*r2[i]*(pBest[i] - self.position[i])


#-----------------------------------------Main Function---------------------------------------------------

E = 0.00001       #Error

#Choose Objective function

function = input("Choose Objective Function: A (Ackley) B(Booth), R(Rastrigin), S(Sphere), E(Easom): ")

#Set diagram title
if function=='B' or function=='b':
    plot_title = "Booth"
elif function=='A' or function == 'a' :                                           
    plot_title = "Ackley"
elif function == 'S' or function == 's':
    plot_title = "Sphere"
elif function == 'R' or function == 'r':
    plot_title = "Rastrigin"
elif function == 'E' or function == 'e':
    plot_title = "Easom"


#Set number of dimensions

if function == 'R' or function == 'r' or function == 'S' or function == 's':
    dimensions = int(input("Give number of dimensions: "))
    plot_title = plot_title +' ' +str(dimensions)+'D'
else:                                                                                      
    dimensions = 2

#Define Prey-Predator ratio
ratio = [4,1]

if dimensions <= 2:
    D = 5
elif dimensions<15:         #Size of predators swarm
    D = 8                   #according to dimensions
else:
    D = 20


F = D*ratio[0]              #Size of preys swarm according to ratio
S = D+F                     #Total number of particles

#Set boundaries according to objective function. U upper boundary, L lower boundary

if function=='B' or function=='b':
    U = [10,10]
    L = [-10,-10]
elif function=='A' or function == 'a' :                                           
    U = [5,5]
    L = [-5,-5]
elif function == 'S' or function == 's':
    U = [10 for i in range(dimensions)]
    L = [-10 for i in range(dimensions)]
elif function == 'R' or function == 'r':
    U = [5.12 for i in range(dimensions)]
    L = [-5.12 for i in range(dimensions)]
elif function == 'E' or function == 'e':
    U = [100,100]
    L = [-100,-100]

#Set optimal solution according to objective function

optimal = [0]*dimensions

if function == 'B' or function == 'b':                    
    optimal = [1,3]
elif function == 'E' or function == 'e':
    optimal = [m.pi,m.pi]

#Set maximum velocity limit

vmax = [0]*dimensions
for i in range(dimensions):
    vmax[i] = m.sqrt(dimensions*(U[i] - L[i]))/10

Wmin = 0.4          #Minimum inertia rate value
Wmax = 0.9          #Maximum inertia rate value
dfMax = 4           #Maximum Distance Coefficients value
dfMin = 2           #Minimum Distance Coefficients value
it = 500            #Maximum number of iterations

#------------------Algorithm------------------------
rd.seed(t.process_time())
                
predators = []              #List of predator's particles
preys = []                  #List of prey's particles

#Set initial particles position
positions = []
for i in range(S):
    temp = [rd.uniform(L[i],U[i]) for i in range(dimensions)]
    positions.append(temp)
                
#Set initial particles velocity
velocities = []
for i in range(S):
    temp = [rd.uniform(-vmax[i],vmax[i]) for i in range(dimensions)]
    velocities.append(temp)

#Divide particles into predators and preys swarms
temp = 0
temp1 = 0

for i in range(S):
    a = rd.randint(0,1)
    if (a == 0 and temp !=D) or temp1==F:
        aPredator = Predator(positions[i],velocities[i])
        predators.append(aPredator)                             #Create Predators Particles and swarm
        temp +=1

    elif (a == 1 and temp1!=F) or temp==D:
        aPrey = Prey(positions[i],velocities[i])
        preys.append(aPrey)                                     #Create Preys Particles and swarm
        temp1 +=1

predatorsBest = [0]*dimensions      #Predators swarm best
preysBest = [0]*dimensions          #Preys swarm best
globalBest = [0]*dimensions         #Global swarm best

predatorsBestFitness = 0            #Predators swarm best fitness value
preysBestFitness = 0                #Preys swarm best fitness value
        
bestValues = []                     #Best value of objective function for every iteration, used for diagram
iterations = []                     #Number ofiterations the algorithm ran, used for diagram
    
for i in range(it):                
    #Calculate Predators swarm best and and Fitness value for each predator
    for j in range(D):
        predators[j].evaluate()
        predators[j].Best()
        if j == 0:
            predatorsBest = predators[j].best[:]
            predatorsBestFitness = predators[j].bestFitness
        elif predators[j].bestFitness < predatorsBestFitness:   
            predatorsBest = predators[j].best[:]
            predatorsBestFitness = predators[j].bestFitness

    #Calculate Preys swarm best and Fitness value for each prey
    for j in range(F):
        preys[j].evaluate()
        preys[j].Best()
        if j == 0:
            preysBest = preys[j].best[:]
            preysBestFitness = preys[j].bestFitness
        elif preys[j].bestFitness < preysBestFitness:
            preysBest = preys[j].best[:]
            preysBestFitness = preys[j].bestFitness

    #Calculate Global best
    globalBest = predatorsBest[:]
    if preysBestFitness < predatorsBestFitness:                 
        globalBest = preysBest[:]

    #Calculate Inertia Rate
    w = Wmax - ((Wmax-Wmin)/it)*i

    #Calclulate fear propability
    pf = 0.99**i

    #Calculate distance coefficient a
    a = 0.99**i

    #Calculate distance coefficient b
    b = dfMin + ((dfMax-dfMin)/it)*i
                
    #Predators Velocity and Position Update    
    for j in range(D):
        predators[j].setVelocity(preysBest)                     
        predators[j].fixVelocity()
        predators[j].setPosition()
        predators[j].fixPosition()

    #Preys Velocity and Position Update
    for j in range(F):
        d = preys[j].EuclideanDistance(predators)              
        preys[j].setVelocity(preysBest,d,a,b,w,pf)
        preys[j].fixVelocity()
        preys[j].setPosition()
        preys[j].fixPosition()

    bestValues.append(f(globalBest))                            
    iterations.append(i)
            
    #Calculate distance between optimal known solution and current best solution

    distance = abs(f(globalBest) - f(optimal))                  

    #Test convergence criteria
    if distance < E:
        break
    
#Best position and best position value
print("Global Best:", globalBest,"\nGlobal Best value:",f(globalBest))
 
#Plot diagram where X = iteratons and Y = f(bestValue)

X = np.array(iterations)
Y = np.array(bestValues)

plt.plot(X,Y)
plt.xlabel("Iterations")                                
plt.ylabel("f Optimal Solution")
plt.title(plot_title)
plt.grid()
plt.show()
