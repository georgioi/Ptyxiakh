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
    elif function=='A' or function == 'a' :
        result = -20 * m.exp(-0.2*m.sqrt(0.5*(x[0]**2+x[1]**2))) - m.exp(0.5*(m.cos(2*m.pi*x[0]) + m.cos(2*m.pi*x[1]))) + m.e + 20 #Ackley
    elif function == 'S' or function == 's':
        for i in range(dimensions):
            result = result + x[i]**2   #Sphere
    elif function == 'R' or function == 'r':
        for i in range (dimensions):
            result = result + (x[i]**2 - 10*m.cos(2*m.pi*x[i]))  #Rastrigin   
        result = result + 10*dimensions
    elif function == 'E' or function == 'e':
        result = -m.cos(x[0])*m.cos(x[1])*m.exp(-(x[0]-m.pi)**2 - (x[1]-m.pi)**2)   #Easom
    return result


#Particles Class

class Particles():

    #Constructor that initializes particle's position and velocity
    #position and velocity parameters are initital particle's position and velocity
    def __init__(self):
        self.position = [rd.uniform(L[i],U[i]) for i in range(dimensions)]          #Particle's position
        self.velocity = [rd.uniform(-vmax[i],vmax[i]) for i in range(dimensions)]   #Particle's velocity
        self.best = self.position[:]                                                #Best self position
        self.positionFitness = f(self.position)                                     #Current position fitness
        self.bestFitness = f(self.best)                                             #Best self position fitness


    #Check if current position is personal best
    def Best(self):
        if self.positionFitness < self.bestFitness:
            self.best  = self.position[:]
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

#Carnivores Class. This is subclass of particles class
                
class Carnivore(Particles):

    #Constructor that calls base's class constructor.
    def __init__(self):
        super().__init__()
        
    #Update Carnivore's Velocity according to formula. swarmBest Parameter is the best position of carnivores swarm
    def setVelocity(self,swarmBest):
        r = [rd.uniform(0,1) for i in range(dimensions)]         #Random variable used to update carnivores velocity
        for i in range(dimensions):
            self.velocity[i] = self.velocity[i] + r[i]* (swarmBest[i] - self.position[i])
        
#Omnivores Class. This is subclass of particles class
            
class Omnivore(Particles):
    def __init__(self):
        super().__init__()
        self.c1 = 2                 #Cognitive Coefficient
        self.c2 = 2                 #Social Coefficient
        self.c3 = 2                 #Carnivores Coefficient

    #Find the nearest particle element from predators list to self omnivore
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

    #Calculate the measure of the effect that other omnivores or carnivores have to self omnivore. d parameter is distance from nearest omnivore or carnivore. a and b parameters are distance coefficients.
    def D(self,d,a,b):
        return a*m.exp(-b*d)

    #Propability of omnivore being a predator
    def predatorPropability(self,do,dc):
        if not do + dc == 0:
            self.p = do/(dc+do)
        else:
            self.p = 0.5

    #Calculate omnivore velocity according to formula. gbest parameter is omnivores swarm best position. do and dc parameters are self distance from nearset omnivore and carnivore respectively
    #Pfoc is fear factor of omnivore to carnivore. w parameter is inertia rate. a and b are parameters used for function D.
    def setVelocity(self,gBest,do,dc,a,b,w):
        r1 = [rd.uniform(0,1) for i in range(dimensions)]    
        r2 = [rd.uniform(0,1) for i in range(dimensions)]        #r1, r2, r3 and r are random variables 
        r3 = [rd.uniform(0,1) for i in range(dimensions)]        #used to update omnivores' velocity
        r  = [rd.uniform(0,1) for i in range(dimensions)]
        
        Pfoc = 1 - dc/PfocMin
        for i in range(dimensions):
            self.velocity[i] =(1-self.p)*(w*self.velocity[i] + self.c1*r1[i]*(self.best[i] - self.position[i]) + self.c2*r2[i]*(gBest[i] - self.position[i]) + Pfoc*self.c3*r3[i]*self.D(do,a,b))
            + self.p*r[i]*(gBest[i] - self.position[i])


#Omnivores Class. This is subclass of particles class

class Herbivore(Particles):
    def __init__(self):
        super().__init__()
        self.c1 = 2             #Cognitive Coefficient
        self.c2 = 2             #Social Coefficient
        self.c3 = 2             #Omnivores Coefficient
        self.c4 = 2             #Carnivores Coefficient
        
               
    #Find the nearest particle element from predators list to self herbivore
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

    #Calculate the measure of the effect that omnivores or carnivores have to the herbivore. d parameter is distance from nearest omnivore or carnivore. a and b parameters are distance coefficients.
    def D(self,d,a,b):
        return a*m.exp(-b*d)        
        
    #Calculate herbivore velocity according to formula. gbest parameter is herbivores swarm best position. do and dc parameters are self distance from nearset omnivore and carnivore respectively
    #Pfho and Pfhc are fear factors of herbivore to omnivore and carnivore respectively.w parameter is inertia rate
    def setVelocity(self,gBest,dc,do,a,b,w):
        r1 = [rd.uniform(0,1) for i in range(dimensions)]    
        r2 = [rd.uniform(0,1) for i in range(dimensions)]    #r1, r2, r3 and r4 are random variables 
        r3 = [rd.uniform(0,1) for i in range(dimensions)]    #used to update herbivores velocity
        r4 = [rd.uniform(0,1) for i in range(dimensions)]

        Pfho = 1 - do/PfhoMin
        Pfhc = 1 - dc/PfhcMin
        for i in range(dimensions):
            self.velocity[i] =  w*self.velocity[i] + self.c1*r1[i]*(self.best[i] - self.position[i]) + self.c2*r2[i]*(gBest[i] - self.position[i]) + Pfho*self.c3*r3[i]*self.D(do,a,b) + Pfhc*self.c4*r4[i]*self.D(dc,a,b)

#-----------------------------------------Main Function---------------------------------------------------

E = 0.00001      #Error

#Choose Objective function

function = input("Choose Objective Function: A (Ackley) B(Booth), R(Rastrigin), S(Sphere),E(Easom): ")

#Set number of dimensions

if function == 'R' or function == 'r' or function == 'S' or function == 's':
    dimensions = int(input("Give number of dimensions: "))
else:                                                                                      
    dimensions = 2

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
    vmax[i] = m.sqrt(dimensions*(U[i] - L[i]))

space = 0
for i in range(dimensions):
    space = space + (U[i] - L[i])**2
space = m.sqrt(space)

PfhcMin = (space)/70  #Minimum distance for a herbivore to start fear a carnivore
PfocMin = (space)/73  #Minimum distance for an omnivore to start fear a carnivore  
PfhoMin = (space)/74  #Minimum distance for a herbivore to start fear an omnivore

#Choose Enviroment (wild, average or calm)
e = input("Choose Enviroment: Wild(W), Average(A), Calm(C): ")

#Define herbivore-omnivore-carnivore ratio according to enviroment

if e == 'W' or e == 'w':
    ratio = [10,3,1]
elif e == 'A' or e == 'a':
    ratio = [25,6,1]
elif e == 'C' or e == 'c':
    ratio = [40,10,1]
    
#Size of carnivores swarm according to dimensions and enviroment

if dimensions <= 2:
    if e == 'C' or e == 'c':
        nc = 1
    elif e == 'A' or e == 'a':
        nc = 2
    elif e == 'W' or e == 'w':
        nc = 4
elif dimensions<15:         
    if e == 'C' or e == 'c':
        nc = 2
    elif e == 'A' or e == 'a':
        nc = 3
    elif e == 'W' or e == 'w':
        nc = 7                 
else:
    if e == 'C' or e == 'c':
        nc = 3
    elif e == 'A' or e == 'a':
        nc = 5
    elif e == 'W' or e == 'w':
        nc = 11

nh = nc*ratio[0]            #Size of herbivores swarm according to ratio        
no = nc*ratio[1]            #Size of omnivores swarm according to ratio

Wmin = 0.4          #Minimum inertia rate value
Wmax = 0.9          #Maximum inertia rate value
dfMax = 2           #Maximum Distance Coefficients value
dfMin = 1           #Minimum Distance Coefficients value
it = 1500           #Maximum number of iterations

#------------------Algorithm------------------------
table = []
table1 = []
table2 = []
for u in range(1):
    start = t.perf_counter()
    rd.seed(t.process_time())

    carnivores = []                             #List of carnivore's particles
    omnivores = []                              #List of omnivores's particles
    herbivores = []                             #List of herbivores's particles

    for i in range(nc):
        aCarnivore = Carnivore()                #Create carnivores particles and swarm
        carnivores.append(aCarnivore)
                
    for i in range(no):
        anOmnivore = Omnivore()
        omnivores.append(anOmnivore)            #Create omnivores particles and swarm
                
    for i in range(nh):
        aHerbivore = Herbivore()
        herbivores.append(aHerbivore)           #Create herbivores particles and swarm

    carnivoresBest = [0]*dimensions             #Carnivores swarm best                
    herbivoresBest = [0]*dimensions             #Herbivores swarm best
    omnivoresBest = [0]*dimensions              #Omnivores swarm best
    globalBest = [0]*dimensions                 #Global swarm best

    carnivoresBestFitness = 0                   #Carnivores swarm best fitness value
    herbivoresBestFitness = 0                   #Herbivores swarm best fitness value
    omnivoresBestFitness = 0                    #Omnivores swarm best fitness value

    bestValues = []                             #Best values of objective function for every iteration, used fot diagram
    iterations = []                             #How many iterations the algorithm run, used fot diagram

    for i in range(it):
                
        #Calculate Carnivores swarm best and Fitness value for each carnivore
        for j in range(nc):
            carnivores[j].evaluate()
            carnivores[j].Best()
            if j==0:
                carnivoresBest = carnivores[j].best[:]
                carnivoresBestFitness = carnivores[j].bestFitness
            elif carnivores[j].bestFitness < carnivoresBestFitness:         
                carnivoresBest = carnivores[j].best[:]
                carnivoresBestFitness = carnivores[j].bestFitness

        #Calculate Herbivores swarm best and Fitness value for each herbivore       
        for j in range(nh):
            herbivores[j].evaluate()
            herbivores[j].Best()
            if j==0:
                herbivoresBest = herbivores[j].best[:]
                herbivoresBestFitness = herbivores[j].bestFitness
            elif herbivores[j].bestFitness < herbivoresBestFitness:         
                herbivoresBest = herbivores[j].best[:]
                herbivoresBestFitness = herbivores[j].bestFitness
                        
        #Calculate Omnivores swarm best and Fitness value for each omnivore
        for j in range(no):
            omnivores[j].evaluate()
            omnivores[j].Best()
            if j==0:
                omnivoresBest = omnivores[j].best[:]
                omnivoresBestFitness = omnivores[j].bestFitness
            elif omnivores[j].bestFitness < omnivoresBestFitness:           
                omnivoresBest = omnivores[j].best[:]
                omnivoresBestFitness = omnivores[j].bestFitness

        #Calculate Global best
        globalBest = carnivoresBest[:]

        if herbivoresBestFitness < carnivoresBestFitness:
            globalBest = herbivoresBest[:]                                                                     
                
        if omnivoresBestFitness < herbivoresBestFitness and omnivoresBestFitness < carnivoresBestFitness:
            globalBest = omnivoresBest[:]
      
        #Calculate Inertia Rate
        w = Wmax - ((Wmax-Wmin)/it)*i

        #Calculate distance coefficient a
        a = 0.99**i

        #Calculate distance coefficient b
        b = dfMin + ((dfMax-dfMin)/it)*i

        #Carnivores Velocity and Position update
        for j in range(nc):
            carnivores[j].setVelocity(carnivoresBest)                       
            carnivores[j].fixVelocity()
            carnivores[j].setPosition()
            carnivores[j].fixPosition()


        #Herbivores Velocity and Position update
        for j in range(nh):
            dc = herbivores[j].EuclideanDistance(carnivores)                
            do = herbivores[j].EuclideanDistance(omnivores)             
            herbivores[j].setVelocity(herbivoresBest,dc,do,a,b,w)
            herbivores[j].fixVelocity()
            herbivores[j].setPosition()
            herbivores[j].fixPosition()

        #Omnivores Velocity and Position update
        for j in range(no):
            temp = omnivores[:]
            temp.pop(j)
            do = omnivores[j].EuclideanDistance(temp)
            dc  = omnivores[j].EuclideanDistance(carnivores)
            omnivores[j].predatorPropability(do,dc)                         
            omnivores[j].setVelocity(omnivoresBest, do,dc,a,b,w)
            omnivores[j].fixVelocity()
            omnivores[j].setPosition()
            omnivores[j].fixPosition()

        bestValues.append(f(globalBest))
        iterations.append(i)

        #Calculate distance between optimal known solution and current best solution      
        distance = abs(f(globalBest)-f(optimal))
        
        #Test convergence criteria
        if distance < E:
            break
        
    end = t.perf_counter()

    table.append(f(globalBest))
    table1.append(i+1)
    table2.append(end-start)

#Best position and best position value
print("Global Best:", globalBest,"\nGlobal Best value:",f(globalBest))

#Plot diagram where X = iteratons and Y = f(bestValue)

X = np.array(iterations)
Y = np.array(bestValues)

plt.plot(X,Y)
plt.xlabel("Iterations")
plt.ylabel("f Optimal Solution")
plt.title("Ackley Calm")
plt.grid()
plt.show()
    
#-------------------For tests---------------------
average = sum(table)/50
averageIt = sum(table1)/50
dif = [abs(i - f(optimal)) for i in table]
min_index = dif.index(min(dif))
max_index = dif.index(max(dif))
best = table[min_index]
worst = table[max_index]
averageDur = sum(table2)/50

file = []
file.append("\tSphere 30D Wild")
file.append("Average Solution: "+str(average))
file.append("Best Solution: "+str(best))
file.append("Worst Solution: "+str(worst))
file.append("Average number of Iterations: "+str(averageIt))
file.append("Maximum Iterations: "+str(it))
file.append("Average Duration: "+str(averageDur)+"\n")
    
f = open("2ο Σετ.txt","a")
for line in file:
    f.write(line+"\n")
f.close()
