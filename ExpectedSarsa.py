import blackjack
from pylab import *

numEpisodes = 2000

returnSum = 0.0
for episodeNum in range(numEpisodes):
	G = 0
	currentstate = blackjack.init()
	while(currentstate != -1):
		action = randint(2) #randomly pick the action
		next = blackjack.sample(currentstate, action)
		G = G + next[0]
		currentstate = next[1]
	print "Episode: ", episodeNum, "Return: ", G
	returnSum = returnSum + G
print "Average return: ", returnSum/numEpisodes
