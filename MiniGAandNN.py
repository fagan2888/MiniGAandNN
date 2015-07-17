import random, numpy as np
params = [10, 0.20, 100, 2, 5] # [Init pop (pop=100), mut rate (=5%), num generations (250), chromosome/solution length (2), # winners/per gen]
target = np.array([0.1, 0.9]) #ideal solution, make
curPop = np.random.random_sample((params[0], params[3])) #current population, initialize to random
nextPop = np.zeros((params[0], params[3]))
fitVec = np.zeros((params[0], 2)) #1st col is indices
for i in range(params[2]): #iterate through num generations
	fitVec = np.array([np.array([x, np.sum((curPop[x] - target)**2)]) for x in range(params[0])]) #Create vec of all squared errors
	print("(Gen: #%s) Total error: %s" % (i, np.sum(fitVec[:,1])))
	winners = np.zeros((params[4], params[3])) #20x2
	for n in range(len(winners)): #for n in range(10)
		selected = np.random.choice(range(len(fitVec)), 2, replace=False)
		wnr = np.argmin(fitVec[selected,1])
		winners[n] = curPop[int(fitVec[selected[wnr]][0])]
	nextPop[:len(winners)] = winners #populate new gen with winners
	nextPop[len(winners):] = np.array([np.array(np.random.permutation(np.repeat(winners[:, x], ((params[0] - len(winners))/len(winners)), axis=0))) for x in range(winners.shape[1])]).T #Populate the rest of the generation
	nextPop = np.array([np.multiply(1+np.random.normal(0,0.2,2), nextPop[x]) if random.random() < params[1] else nextPop[x] for x in range(nextPop.shape[0])]) #mutate 5% of the population
	curPop = nextPop
print("Best Sol'n:\n%s" % (curPop[np.argmin(fitVec[:,1])],))