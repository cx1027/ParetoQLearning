__author__ = 'X'

import numpy as np
import sys, random
from enum import Enum

# useful utilities
def debug(s):
    if DEBUG:
       print (s)

def enum(*args):
    g = globals()
    i = 0
    for arg in args:
        g[arg] = i
        i += 1
    return i

# Macros
# NUM_ACTIONS = Enum('ACTION_NORTH',
#                    'ACTION_EAST',
#                    'ACTION_SOUTH',
#                    'ACTION_WEST',)

class NUM_ACTIONS(Enum):
    ACTION_NORTH = 1
    ACTION_EAST =2,
    ACTION_SOUTH = 3,
    ACTION_WEST = 4


DEBUG = False



class QAction(object):
    action = None # action is from NUM_ACTIONS
    pareto = [] # store (1,99),(2,98) etc.
    R = None
    def __init__(self, rInit = None, qInit = None, direction = None):
        self.action = direction
        # self.pareto = np.array[(0,0)] 'except finalState'
        self.pareto = []
        self.pareto.append(qInit)
        self.R = rInit



# class QClass(object):
#     """
#     each Qclass has four QAction
#     """
#     Actions = {}
#
#     def __init__(self, rInit = None, qInit = None):
#         for action in NUM_ACTIONS:
#             self.Actions[action] = QAction(action, qInit)
#         self.R = rInit



#learning tables
Q={}
trash ={}


Config = {}

GRID_x = 3
GRID_y = 2
Q_INIT = np.array([0, 0])
R_INIT = np.array([0, 0])

# learning parameters
GAMMA = 0.5

FinalCount = 0

#globals
grid = []
curPos = (0,0)
actions = 0

# remembered information
prevState = None
prevAction = None
prevReward = None

#read grid from file
RewardGrid=[[0,0,0],
          [1,0,0],
          [5,2,0],
          [5,5,3]]


OpenPos = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))

RewardPos = [(1,0),(2,1),(3,2)]


# print grid
def printGrid(Grid):
    print(Grid)
    # for i in Grid:
    #     for j in i:
    #         sys.stdout.write(str(j) + ' ')
    #         sys.stdout.write('\n')




# take the specified action: returns the reward for that action
def takeAction(action):
    global curPos, treasure, grid, actions, prevState, prevAction
    x, y = curPos
    #mark previous State
    prevState = curPos
    prevAction = action

    reward = np.array([0,0])
    try:
        if action == NUM_ACTIONS.ACTION_EAST:
            if y < GRID_y:
                y += 1
        elif action == NUM_ACTIONS.ACTION_NORTH:
            if x > 0:
                x -= 1
        elif action == NUM_ACTIONS.ACTION_WEST:
            if y > 0:
                y -= 1
        elif action == NUM_ACTIONS.ACTION_SOUTH:
            if x < GRID_x:
                x += 1
        #if 到边界，there is no reward??????!!!!!
        curPos = (x, y)
        #if curPos == (0, 0) and trashes == 0 and reward != REWARD_PICKUP:
        reward = np.array([-1, RewardGrid[x][y]])

        actions += 1
    except Exception as e:
        print(e)
        print('x,y:', x, y)
    return reward


def getInitialValuesByPosition(curPos):
    x, y = curPos
    if curPos in RewardPos:
        return (np.array([-1, RewardGrid[x][y]]), None)
    else:
        return (R_INIT, Q_INIT)


# returns the paretp Q-value for an action in the specified state
def paretoQ(curPos, currentParento = []):
    allValues = []
    pareto = []
    #传进来有值
    if len(currentParento) != 0:
        allValues = currentParento
    #传进来没有值
    else:
        for curAction in NUM_ACTIONS:
            key = (curPos, curAction)
            #print('paretoQ--------CurPos:',curPos, ' Action:', curAction)
            if not key in Q:
                (ri, qi) = getInitialValuesByPosition(curPos)
                Q[key] = QAction(ri, qi, curAction)
                ###########################
            # if action == None:
            #     action = curAction
            else:
                #get pareto from each direction in  Q[key] (QClass)
                if len(Q[key].pareto) != 0:
                    allValues.extend(Q[key].pareto)
    #pareto排序
    # print("All pareto values:", allValues)
    b_add = False
    iLen = len(allValues)
    if iLen != 0:
        for i in range(0, iLen):
            jLen = len(pareto)
            if jLen == 0:   # if pareto is empty
                pareto.append(allValues[i])
                continue
            lastj = jLen - 1
            for j in range(0, jLen):
                if jLen >= 1 and j <= jLen - 1 :
                    # print("i, j: ", i, j )
                    a, b = allValues[i]
                    y, z = pareto[j]
                    # print("a,b:", a," ", b, " y, z:",y," ", z, " exiting pareto:", pareto )
                    if (a > y and b > z) :
                        if lastj == j:
                            b_add = True
                        pareto.pop(j)
                        jLen = len(pareto)
                        lastj = len(pareto) - 1
                    elif (a > y and b == z) or (b > z and a == y) :
                        if lastj == j:
                            b_add = True
                        pareto.pop(j)
                        jLen = len(pareto)
                        lastj = len(pareto) - 1
                        continue
                    elif (a > y and b < z) or (b > z and a < y) :
                        if lastj == j:
                            b_add = True
                        continue
                    elif (a == y and b == z) or (a < y and b < z) or (a == y and b < y ) or (a < y and b == z):
                        b_add = False
                        break
                    else:
                        # print("ignore i")
                        continue
            #end of comparing item in pareto, check if need to add i into pareto
            if b_add == True:
                pareto.append(allValues[i])
    #print("Sorted pareto:", pareto)
    return pareto

def paretoQturple(currentParento):

    allValues = currentParento #currentParento is a tuple (action , np.array([num1,num2]))
    pareto = []

    #pareto排序
    # print("All pareto values:", allValues)
    b_add = False
    iLen = len(allValues)
    if iLen != 0:
        for i in range(0, iLen):
            jLen = len(pareto)
            if jLen == 0:   # if pareto is empty
                pareto.append(allValues[i])
                continue # continue for, start i = 1
            lastj = jLen - 1
            for j in range(0, jLen):
                if jLen >= 1 and j <= jLen - 1 :
                    # print("i, j: ", i, j )
                    action_i, value_i = allValues[i]
                    a, b = value_i
                    action_j, value_j = pareto[j]
                    y, z = value_j
                    # print("a,b:", a," ", b, " y, z:",y," ", z, " exiting pareto:", pareto )
                    if (a > y and b > z) :
                        if lastj == j:
                            b_add = True
                        pareto.pop(j)
                        jLen = len(pareto)
                        lastj = len(pareto) - 1
                    elif (a > y and b == z) or (b > z and a == y) :
                        if lastj == j:
                            b_add = True
                        pareto.pop(j)
                        jLen = len(pareto)
                        lastj = len(pareto) - 1
                        continue
                    elif (a > y and b < z) or (b > z and a < y) :
                        if lastj == j:
                            b_add = True
                        continue
                    elif (a == y and b == z) or (a < y and b < z) or (a == y and b < y ) or (a < y and b == z):
                        b_add = False
                        break
                    else:
                        # print("ignore i")
                        continue
            #end of comparing item in pareto, check if need to add i into pareto
            if b_add == True:
                pareto.append(allValues[i])
    #print("Sorted pareto:", pareto)
    return pareto

def getPareto(curPos):
    pareto = []
    if curPos not in RewardPos:
        for curAction in NUM_ACTIONS:
            key = (curPos, curAction)
            if key in Q:
                #get pareto from each direction in  Q[key] (QClass)
                pareto.extend(Q[key].pareto)
    else:
        x, y = curPos
        pareto.append(np.array([-1, RewardGrid[x][y]]))
    return pareto

def getdirectionPareto(curPos, action):
    pareto = []
    if curPos not in RewardPos:
        key = (curPos, action)
        if key in Q:
            #get pareto from each direction in  Q[key] (QClass)
            pareto.extend(Q[key].pareto)
    else:
        x, y = curPos
        pareto.append(np.array([-1, RewardGrid[x][y]]))
    return (action, pareto)

# returns the current reward for exploring further
# def exploration(utility, frequency):
#     if frequency < N: # a num between 0 and 1
#         return REWARD_SUCCESS
#     else:
#         return 'choose an action random'

# determines the best action to take in the current state


def selectBestAction(position):
    bestAction = None
    directionParentoCounts = {} #store count for each direction

    e = random.random()
    if e < 0.3:
        bestAction = random.choice(list(NUM_ACTIONS))
    else:
        for action in NUM_ACTIONS:
            key = (position, action)
            if not key in Q:
                (ri, qi) = getInitialValuesByPosition(position)
                Q[key] = QAction(ri, qi, action)
            else:
                xkey = action
                directionParentoCounts[xkey] = len(Q[key].pareto)
        if(bool(directionParentoCounts)):
            bestAction = max(directionParentoCounts, key = directionParentoCounts.get)
        else:
            bestAction = random.choice(list(NUM_ACTIONS))

    return bestAction

def updateQ(reward, curPos):
    global prevState, prevAction, prevReward, RewardGrid, actions
    # curPos是第一个点
    if prevState == None and prevAction == None:
        return 0

    # '终点为普通点和最后三个点'
    else:
        #R 'later'
        key = (prevState, prevAction) #如果关键字=（a，b）需要先使 key=a，b）
        (x, y) = curPos

        #Q
        #得到当前正方形里的Qvalue
        if curPos not in RewardPos:
            #初始化四个三角
            for action in NUM_ACTIONS:
                currKey = (curPos, action)
                if not currKey in Q:#why if not currKey?????
                    (ri, qi) = getInitialValuesByPosition(curPos)
                    Q[currKey] = QAction(ri, qi, action)

            #计算tempQ=R+γtempQ

            # for action in NUM_ACTIONS:
            #     currKey = (curPos, action) #TODO:to be changed by daciba
            #     if not key in Q:
            #         (ri, qi) = getInitialValuesByPosition(curPos)
            #         Q[currKey] = QAction(ri, qi, action)


            currParento = []
            #calc p for each direction of curr Pos
            for action in NUM_ACTIONS:
                currKey = (curPos, action)
                for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
                    p = Q[currKey].R + GAMMA * p #todo: change key
                    currParento.append(p)

            #得到当前正方形里的pareto
            #赋给三角形
            if not key in Q:#why if not currKey?????
                    (ri, qi) = getInitialValuesByPosition(prevState)
                    Q[key] = QAction(ri, qi, prevAction)
            Q[key].pareto = paretoQ(curPos, currParento)
            #print ('trangle value:', Q[key].pareto)
            #转到下一个位置
            prevState = curPos
        if (x,y) in RewardPos:
            currParento = getPareto(curPos)
            #得到当前正方形里的pareto
            #赋给三角形
            if key not in Q:
                (ri, qi) = getInitialValuesByPosition(prevState)
                Q[key] = QAction(ri, qi)

            currParento.extend(Q[key].pareto)
            Q[key].pareto = paretoQ(curPos, currParento)
            #print ('trangle value:', Q[key].pareto)
            #转到下一个位置

            prevState =  None
            prevAction = None
            prevReward = None
            # Q[key].R = np.array([0, 0]) ......<---error prevState has been reset above!
            #print ("Get treasure:", curPos)


        #在终点
        #'终点计算的问题:和普通点一样'
        #R:calculate R for previous state
        # Q[key].R = prevReward
        if not key in Q:
            (ri, qi) = getInitialValuesByPosition(prevState)
            Q[key] = QAction(ri, qi, key)
        Q[key].R = Q[key].R + (reward - Q[key].R) / actions
        prevReward = Q[key].R
        #print ('Reward:', prevReward, ' Actions:', actions)

        if (x, y) in RewardPos:
            actions = 0

def training1():
    global curPos, RewardGrid, actions,GAMMA, N, FinalCount



    printGrid(RewardGrid)

    ALPHA = 1.0
    GAMMA = 0.5

    print ("ALPHA: ", ALPHA, "\nGAMMA: ",GAMMA, "\nGRID_LENGTH: ",GRID_y, "\nGRID_WIDE: ",GRID_x, )

    #total new start
    # curPos = random.choice(GridState)
    # action = random.choice(list(NUM_ACTIONS)) #reward,state,avgreward
    curPos = (0,0)
    while FinalCount < 30:
        # if FinalCount == 0:
        #     action = NUM_ACTIONS.ACTION_SOUTH
        # else:
        action = selectBestAction(curPos)
        print('Cur Pos:', curPos, ' Action:',action, ' FinalCount:', FinalCount)

        reward = takeAction(action)
        updateQ(reward,curPos)
        if curPos in RewardPos:
            FinalCount  += 1
            printing()
            #curPos = random.choice(OpenPos)
            curPos = (0,0)
            #print ('ramdon pos:', curPos)

def printing():
    for pos in OpenPos:
        for action in NUM_ACTIONS:
            key = (pos, action)
            if key not in Q:
                print("-------Key:", key, " NULL")
            else:
                print("-------Key:\t", key, "\t Q.R:\t", Q[key].R, "\t Q.pareto:\t", Q[key].pareto)


# def testing():
#     FinalCount = 0
#
#     curPos = (0, 0)
#     actionOne = None
#     curPareto = getPareto(curPos)
#     if len(curPareto) !=0:
#         target = random.choice(curPareto)
#
#     while len(curPareto) == 0:
#         curPos = random.choice(OpenPos)
#         curPareto = getPareto(curPos)
#         if len(curPareto) !=0:
#             target = random.choice(curPareto)
#
#
#
#     for action in NUM_ACTIONS:
#         key = (curPos, action)
#         if key in Q:
#             for pa in Q[key].pareto:
#                 if np.equal( pa , target).all():
#                    actionOne = action
#     takeAction(actionOne)
#
#     while FinalCount < 10:
#          for action in NUM_ACTIONS:
#              key = (curPos,action)
#
#              for p in getPareto(curPos):
#                  print('Cur Pos:', curPos, ' Action:',action, ' FinalCount:', FinalCount)
#                  if np.equal(GAMMA * p + Q[key].R ,target).all():
#
#                      reward = takeAction(action)
#                      target = p
#                      if curPos in RewardPos:
#                          FinalCount += 1
#                          curPos = random.choice(OpenPos)
#              print("Key:", key)


def testing1():
    FinalCount = 0

    curPos = (0, 0)

    # calculate Q
    currentParento=[]
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
            p = Q[currKey].R + GAMMA * p #todo: change key
            currentParento.append((action, p))
    #pareto Q
    print ("currentParento",currentParento)
    sortedParento = paretoQturple(currentParento)
    print ("sortedParento", sortedParento)

    # select Q
    targetTuple = random.choice(sortedParento)
    targetaction = targetTuple[0]
    takeAction(targetaction)
    target = targetTuple[1]
    print("target:", target)

    #loop
    while FinalCount < 3:
        found = False
        foundAction = None
        if curPos in RewardPos:
            FinalCount += 1
            curPos = (0, 0)

        else:
            for action in NUM_ACTIONS:
                key = (curPos,action)
                act, actionPareto = getdirectionPareto(curPos, action)
                for p in actionPareto:
                    print('Cur Pos:', curPos, ' Action:',action, ' FinalCount:', FinalCount)
                    if np.less(GAMMA * p + Q[key].R - target ,np.array([0.001, 0.001])).all():
                        found = True
                        target = p
                        break
                if found:
                    foundAction = action
                    print("########## found action:", action)
                    break

        takeAction(foundAction)#use the found action above where move curPos forward

    print("Key:", key)

def run():
    training1()
    testing1()

run()


