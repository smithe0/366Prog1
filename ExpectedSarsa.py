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
	currentstate = blackjack.init()
	while(currentstate != -1):
	       
            rnumber = random.rand()
	   
            if rnumber < epsilonbehavior:
                action = random.randint(2)
            else:
	        #Pick highest action at state S
                if states[currentstate][0] >= states[currentstate][1]:
                    #hit
	           action = 1
                else:
                    action = 0
	           #stay 	       
	       
            next = blackjack.sample(currentstate, action)
            G = G + next[0]
            greedychance = 1-epsilonbehavior
            highest = argmax(states[next[1]])
            
            target = (greedychance * states[next[1]][highest]) + (epsilonbehavior * (0.5*states[next[1]][0] + 0.5*states[next[1]][1]))
            
            
            states[currentstate][action] = states[currentstate][action] + alpha * (next[0] + target - states[currentstate][action]) 
            
            currentstate = next[1]
            
	#print "Episode: ", episodeNum, "Return: ", G
	returnSum = returnSum + G
	if(episodeNum%10000 == 0):
	    print "Average return ",episodeNum,": ", returnSum/numEpisodes
print "Average return: ", returnSum/numEpisodes
