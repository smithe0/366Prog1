import blackjack
from pylab import *
import numpy as n

numEpisodes = 2000000
learningEpisodes = 1000000

states = 0.00001*n.random.rand(181,2)
epsilonMu = 0.01
epsilonPi = 0.0
alpha = 0.001
discount = 1
epsilon = epsilonMu

#print states


returnSum = 0.0
for episodeNum in range(numEpisodes):
    G = 0
    #Start a new game of blackjack
    currentstate = blackjack.init()
    #Continue this game until the terminal state is reached
    while(currentstate != -1):
        #Get a random number between 0 and 1, if its less than epsilon behavior, then explore

        if episodeNum >= learningEpisodes: 
            epsilon = epsilonPi
            alpha = 0
        
        rnumber = n.random.rand()
        if rnumber < epsilon:
            action = n.random.randint(2)
        else:
	    #If not exploring, pick the highest action at state S
            action = argmax(states[currentstate])       
	
	#Get the next state, get reward and next state
        next = blackjack.sample(currentstate, action)
        reward = next[0]
        nextstate = next[1]
        #Add to return
        G = G + reward
        
        #Get chance of being greedy
        greedychance = 1-epsilon
        
        #Get best value at the next state
        highest = argmax(states[nextstate])
        
        #Expected sarsa calculation (greedy * best_next_state_action) + (explore * (0.5*next_state_action1 + 0.5*next_state_action2))
        target = (greedychance * states[nextstate][highest]) + (epsilon * (0.5*states[nextstate][0] + 0.5*states[nextstate][1]))
        states[currentstate][action] = states[currentstate][action] + alpha * (reward + target - states[currentstate][action]) 
            
        currentstate = nextstate
            
	#print "Episode: ", episodeNum, "Return: ", G
    returnSum = returnSum + G
    if(episodeNum%10000 == 0 and episodeNum != 0):
	print "Average return ",episodeNum,": ", returnSum/episodeNum


def returnPolicy(self): return n.argmax(states[self]) #This took forever to come up with, but why does this work???? wtf?

print "Average return: ", returnSum/numEpisodes
#Print the policy
blackjack.printPolicy(returnPolicy)
