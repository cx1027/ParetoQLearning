__author__ = 'X'

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:

        if maxY:
            if(pair[1] >= p_front[-1][1]):
                if pair[0] == p_front[-1][0] and pair[1] == p_front[-1][1]:
                    p_front.pop(-1) #both equal

                p_front.append(pair)
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    print("x:",p_frontX,"y:",p_frontY)
    return p_frontX, p_frontY


import matplotlib.pyplot as plt
import numpy as np
data = np.array([[1,1]
                ,[2,1]
                ,[5,2]
                ,[5,2]
                ,[-1,8]
                ,[1,1]
                ,[2,4]
                ,[2,4]
                ,[1,7]
                ,[0,0],[0,0],[0,0],[0,0],[0,0]
                 ])
# data = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
Xs= data[:,0]
Ys = data[:,1]
print("Xs:",Xs,"\nYs:",Ys)
# Find lowest values for cost and highest for savings
p_front = pareto_frontier(Xs, Ys, maxX = True, maxY = False)#True)
# Plot a scatter graph of all results
plt.scatter(Xs, Ys)
# Then plot the Pareto frontier on top
plt.plot(p_front[0], p_front[1])
plt.show()