import blackjack
from pylab import *
from numpy import *

numEpisodes = 1000000

states = 0.00001*rand(181,2)
epsilonbehavior = 0.01
epsilontarget = 0.01
alpha = 0.001
discount = 1

#print states


returnSum = 0.0
for episodeNum in range(numEpisodes):
    G = 0
    #Start a new game of blackjack
    currentstate = blackjack.init()
    
    #Continue this game until the terminal state is reached
    while(currentstate != -1):
        #Get a random number between 0 and 1, if its less than epsilon behavior, then explore
        rnumber = random.rand()
        if rnumber < epsilonbehavior:
            action = random.randint(2)
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
        greedychance = 1-epsilonbehavior
        
        #Get best value at the next state
        highest = argmax(states[nextstate])
        
        #Expected sarsa calculation (greedy * best_next_state_action) + (explore * (0.5*next_state_action1 + 0.5*next_state_action2))
        target = (greedychance * states[nextstate][highest]) + (epsilonbehavior * (0.5*states[nextstate][0] + 0.5*states[nextstate][1]))
            
            
        states[currentstate][action] = states[currentstate][action] + alpha * (reward + target - states[currentstate][action]) 
            
        currentstate = nextstate
            
	#print "Episode: ", episodeNum, "Return: ", G
    returnSum = returnSum + G
    if(episodeNum%10000 == 0):
	print "Average return ",episodeNum,": ", returnSum/numEpisodes
print "Average return: ", returnSum/numEpisodes
