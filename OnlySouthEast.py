__author__ = 'X'
#update Pareto
#update testing part

import numpy as np
import sys, random
from enum import Enum
import os, csv,logging
from math import *
import datetime

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

class Object:
   def __init__(self, **attributes):
      self.__dict__.update(attributes)

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
        self.N = N_INIT



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

GRID_x = 10
GRID_y = 9
Q_INIT = np.array([0, 0])
R_INIT = np.array([0, 0])
N_INIT = 0

# learning parameters
GAMMA = 1

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
RewardGrid=[[0,0 ,0  ,0,0,0,0,0,0,0],
            [5,0 ,0  ,0,0,0,0,0,0,0],
            [5,80,0  ,0,0,0,0,0,0,0],
            [5,5 ,120,0,0,0,0,0,0,0],
            [5,5 ,5  ,140,145,150,0,0,0,0],
            [5,5 ,5  ,5  ,5  ,5  ,0,0,0,0],
            [5,5 ,5  ,5  ,5  ,5  ,0,0,0,0],
            [5,5 ,5  ,5  ,5  ,5  ,163,166,0,0],
            [5,5 ,5  ,5  ,5  ,5  ,5  ,5  ,0,0],
            [5,5 ,5  ,5  ,5  ,5  ,5,5,473,0],
            [5,5 ,5  ,5  ,5  ,5  ,5,5,5,1750]
            ]


OpenPos = ((0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),
           (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),
           (2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),
           (3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
           (4,6),(4,7),(4,8),(4,9),
           (5,6),(5,7),(5,8),(5,9),
           (6,6),(6,7),(6,8),(6,9),
           (7,8),(7,9),
           (8,8),(8,9),
           (9,9)
           )

RewardPos = [(1,0),(2,1),(3,2),(4,3),(4,4),(4,5),(7,6),(7,7),(9,8),(10,9)]


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

    reward = np.array([-1,0])
    try:
        if action == NUM_ACTIONS.ACTION_EAST:
            if y < GRID_y:
                y += 1
        elif action == NUM_ACTIONS.ACTION_NORTH:
            if x > 0:
                x -= 1
        elif action == NUM_ACTIONS.ACTION_WEST:
            if y > 0 and curPos !=(5,6) and curPos !=(6,6) and curPos !=(8,8):
                y -= 1
        elif action == NUM_ACTIONS.ACTION_SOUTH:
            if x < GRID_x:
                x += 1
        #if 到边界，there is no reward??????!!!!!
        curPos = (x, y)
        #if curPos == (0, 0) and trashes == 0 and reward != REWARD_PICKUP:
        if curPos in RewardPos:
            reward = np.array([-1, RewardGrid[x][y]])

        #Set N:
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
        return (R_INIT, Q_INIT, N_INIT)


def paretoArray(data):
    """
        pareto all values with action
    :param paretoInAction: array of tuple(array[-1,2], action)
    :return: pareto with action array of tuple([-1,2], action)
    """
    #data = populate(paretoInAction)
    #tuple=[([-1,2], action),([-1,2], action)]
    if isinstance(data[0], tuple):
        # print("source:", data)
        sortedBy1st = sorted(data, key = lambda x: x[0][0], reverse = True) #sort tuple with first value
        # print("sorted:",sortedBy1st)

        pareto = []
        pareto.append(sortedBy1st[0])
        for item in sortedBy1st[1:]:
            if item[0][1] >= pareto[-1][0][1]: #get the last item to compare the 2nd value in array
                #item[left]= p[left]
                if item[0][0] == pareto[-1][0][0] :
                    if item[0][1] > pareto[-1][0][1]:
                        pareto.pop(-1)
                        pareto.append(item)
                #item[right]< p[right]
                elif item[0][1] > pareto[-1][0][1]:
                    pareto.append(item)
        # print("pareto:", pareto)
        return pareto
    #ndarray=[[-1,2], [1,2]]
    if isinstance(data[0], np.ndarray):
        # print("source:", data)
        sortedBy1st = sorted(data, key = lambda x: x[0], reverse = True) #sort tuple with first value
        # print("sorted:",sortedBy1st)

        pareto = []
        pareto.append(sortedBy1st[0])
        for item in sortedBy1st[1:]:
            if item[1] >= pareto[-1][1]: #get the last item to compare the 2nd value in array
                # if item[0] == pareto[-1][0] and item[1] == pareto[-1][1]:
                #     if len(pareto) > 1:
                #         pareto.pop(-1) #both equal
                # if item[1] > pareto[-1][1]:
                #     pareto.append(item)

                #item[left]= p[left]
                if item[0] == pareto[-1][0] :
                    if item[1] > pareto[-1][1]:
                        pareto.pop(-1)
                        pareto.append(item)
                #item[right]< p[right]
                elif item[1] > pareto[-1][1]:
                    pareto.append(item)
        # print("pareto:", pareto)
    return pareto


def populate(data):
    """
        array of actions
        , which each action is a 2d array,
        this function turns array of 2d array to simple array of tuple([-1,1], action]
    :param data: array of tuple(2d array[[-1,2], [0, 3]], action)
    :return:
    """
    if isinstance(data[0], np.ndarray):
        return data
    if isinstance(data[0][0][0], np.ndarray):
        dt = []
        for item in data:
            for p in item[0]:
                dt.append((p,item[1]))
        return dt
    if isinstance(data[0][0][0], int):
        return data

def flatDirectionPareto(data):
    flat = []
    if not (isinstance(data[0], tuple) and isinstance(data[0][0], list) and isinstance(data[0][0][0], np.ndarray)):
        raise ("!!! Expecting data in format: [([[2,1],[1,2]], action, R),(...)].")
    for item in data:
        for ditem in item[0]:
            flat.append((ditem, item[1], item[2]))
    return flat

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
                (ri, qi, ni) = getInitialValuesByPosition(curPos)
                Q[key] = QAction(ri, qi, curAction)
            #get pareto from each direction in  Q[key] (QClass)
            if len(Q[key].pareto) != 0:
                allValues.extend((Q[key].pareto,curAction))
    #pareto排序
    return paretoArray(populate(allValues))

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

def getDirectionPareto(curPos, action):
    """"""
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


def tickoff00(currParentoafter):
    nonzero=[]
    for item in currParentoafter:
        if not np.array_equal(item, np.array([0, 0])):
            nonzero.append(item)
    return nonzero

def updateQ(reward, curPos):
    global prevState, prevAction, prevReward, RewardGrid, actions
    # curPos是第一个点
    if prevState == None and prevAction == None:
        # Q[key].R = Q[key].R + (reward - Q[key].R) / actions
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
                    (ri, qi, ni) = getInitialValuesByPosition(curPos)
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
                # print ("before", Q[currKey].pareto)
                for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
                    p = Q[currKey].R + GAMMA * p #todo: change key
                    currParento.append(p)
                # print ("after", Q[currKey].pareto)
            #得到当前正方形里的pareto
            #赋给三角形
            if not key in Q:#why if not currKey?????
                    (ri, qi, ni) = getInitialValuesByPosition(prevState)
                    Q[key] = QAction(ri, qi, prevAction)
            currParentoafter = paretoQ(curPos, currParento)
            ################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            currParentoafter = tickoff00(currParentoafter)###########################################################
            ####################################################################################################
            if len(currParentoafter) >= 1:
                Q[key].pareto = currParentoafter
            #print ('trangle value:', Q[key].pareto)
            #转到下一个位置
            prevState = curPos

        if (x,y) in RewardPos:
            # if(curPos==(3,2)):
            #   print("3,2")
            currParento = getPareto(curPos)


            #得到当前正方形里的pareto
            #赋给三角形
            if key not in Q:
                (ri, qi, ni) = getInitialValuesByPosition(prevState)
                Q[key] = QAction(ri, qi)

            currParento.extend(Q[key].pareto)
            Q[key].pareto = paretoQ(curPos, currParento)
            #转到下一个位置

            prevState =  None
            prevAction = None
            prevReward = None


        #在终点
        #'终点计算的问题:和普通点一样'
        #R:calculate R for previous state
        # Q[key].R = prevReward
        if not key in Q:
            print("***********key not in Q:", key)
            (ri, qi) = getInitialValuesByPosition(prevState)
            Q[key] = QAction(ri, qi, key)

        Q[key].N += 1
        Q[key].R = Q[key].R + (reward - Q[key].R) / Q[key].N
        # print('key:',key, '\t\tR:', Q[key].R, '=\t\tPR:', prevR, '+ \t(r:' ,reward, '- PR:',prevR, ')/  #:', actions)
        prevReward = Q[key].R
        #print ('Reward:', prevReward, ' Actions:', actions)

        if (x, y) in RewardPos:
            actions = 0


def finalParetoprint(curPos):
    # calculate Q
    currentV=[]
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto=[]
        if currKey in Q:
            print("final Pareto Print: Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
            for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
                p = Q[currKey].R + GAMMA * p #todo: change key
                temppareto.append(p)
        currentV.append((temppareto, action))
    #pareto Q
    print ("currentV",currentV)
    sortedParento = paretoQ(curPos, currentV) # currentParento = ([[0,1],[1,0]], action)
    print ("sortedParento", sortedParento)

def DisplayFinal():
    for Pos in OpenPos:
        print("CurPos:",Pos)
        finalParetoprint(Pos)




def training1():
    global curPos, RewardGrid, actions,GAMMA, N, FinalCount



    printGrid(RewardGrid)

    ALPHA = 1.0
    GAMMA = 1

    print ("ALPHA: ", ALPHA, "\nGAMMA: ",GAMMA, "\nGRID_LENGTH: ",GRID_y, "\nGRID_WIDE: ",GRID_x, )

    #total new start
    # curPos = random.choice(GridState)
    # action = random.choice(list(NUM_ACTIONS)) #reward,state,avgreward
    curPos = (0,0)
    while FinalCount < 5000:

        # if curPos ==(9,9):
        #     print (actions)
        # if FinalCount == 0:
        #     action = NUM_ACTIONS.ACTION_SOUTH
        # else:
        #action = selectBestAction(curPos)
        action = random.choice(list(NUM_ACTIONS))
        # print('Cur Pos:', curPos, ' FinalCount:', FinalCount, ' Action:',action)

        reward = takeAction(action)
        updateQ(reward,curPos)

        # print('*** finalCount:', FinalCount)
        if curPos in RewardPos:
            FinalCount  += 1
            #print("final curPos",curPos)
            # DisplayGrid()
            #curPos = random.choice(OpenPos)
            curPos = (0,0)
            #print ('ramdon pos:', curPos)
    DisplayGrid()
    DisplayFinal()

def DisplayGrid():
    for pos in OpenPos:
        for action in NUM_ACTIONS:
            key = (pos, action)
            if key not in Q:
                print("-------Key:", key, " NULL")
            else:
                print("-------Key:\t", key, "\t Q.R:\t", Q[key].R, "\t Q.pareto:\t", Q[key].pareto)


def getOrignalPareto(caculateQ, tempParetos, R):
    # targetQ = np.array([0,0])
    for q in tempParetos:
        if np.equal(caculateQ , R + GAMMA * q).all():
            targetQ = q
            break
    # if targetQ == None:
    #     raise("Couldn't find orginal Q")
    return  targetQ


def Reset(curPos):
    # calculate Q
    currentV=[]
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto=[]
        # print("Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
        for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
            p = Q[currKey].R + GAMMA * p #todo: change key
            temppareto.append(p)
        currentV.append((temppareto, action))
    #pareto Q
    print ("currentV",currentV)
    sortedParento = paretoQ(curPos, currentV) # currentParento = ([[0,1],[1,0]], action)
    print ("sortedParento", sortedParento)

    # select Q
    targetTuple = random.choice(sortedParento)
    targetAction = targetTuple[1]
    # print("action:",targetAction)
    targetKey = (curPos, targetAction)
    calcTarget = targetTuple[0]
    target = getOrignalPareto(calcTarget, Q[targetKey].pareto,  Q[targetKey].R)

    # print("target:", target)
    return target, targetAction

def Resetinturn(curPos):
    # calculate Q
    currentV=[]
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto=[]
        # print("Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
        for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
            p = Q[currKey].R + GAMMA * p #todo: change key
            temppareto.append(p)
        currentV.append((temppareto, action))
    #pareto Q
    print ("currentV",currentV)
    sortedParento = paretoQ(curPos, currentV) # currentParento = ([[0,1],[1,0]], action)
    length=len(sortedParento)
    print ("sortedParento", sortedParento)

    return sortedParento

    # select Q
    # targetTuple = sortedParento
    # targetAction = targetTuple[1]
    # # print("action:",targetAction)
    # targetKey = (curPos, targetAction)
    # calcTarget = targetTuple[0]
    #target = getOrignalPareto(calcTarget, Q[targetKey].pareto,  Q[targetKey].R)
    #
    # # print("target:", target)
    # return target, targetAction,length

def getMinP(paretos,target):
    """pareto = (np.array([1,1]),Action, R)"""
    calcP = []
    for p in paretos:
        calc = GAMMA * p[0] + p[1]
        calcP.append((((calc - target)**2).sum(), p[2], p[0]))

    return min(calcP, key = lambda item: item[0])

def testing1():
    global curPos, prevAction, prevState, actions
    FinalCount = 0
    curPos = (0, 0)

    target, targetAction = Reset(curPos)
    print("target:",target)
    takeAction(targetAction)

    #loop
    while FinalCount < 200:
        if curPos in RewardPos:
            FinalCount += 1
            print("-----------Reach Reward Pos: ",curPos, "\twith action count:", actions, "\tFinalCount:", FinalCount)
            curPos = (0, 0)
            actions = 0
            target, targetAction = Reset(curPos)
            print("target:",target)

        else:
            ap = []
            for action in NUM_ACTIONS:
                key = (curPos,action)
                a, paretos = getDirectionPareto(curPos, action)
                #print("Action:", key, "\tparetos:", paretos)
                ap.append((paretos, Q[key].R, action))
            apdata = flatDirectionPareto(ap)
            diff, targetAction, target = getMinP(apdata, target)
            #print("Found! diff:",diff, "\tTarget Action:", targetAction, "\tnew target value:", target )

        print("###From Pos:", curPos, "\t take action:", targetAction, "\ttarget:", target)
        takeAction(targetAction)#use the found action above where move curPos forward

    #print("Key:", key)

def testinginturn():
    global curPos, prevAction, prevState, actions, FinalCount
    finalStateCount = 0
    trailCount = 0



    paretoOfFirstPos = Resetinturn((0,0))
    print("pareto in 0,0:",paretoOfFirstPos,"\n number:",len(paretoOfFirstPos))

    ### trail
    while trailCount < runSettings['totalTrailCount']:
    ### finalStateUpperBound
        while finalStateCount < runSettings['finalStateUpperBound']:
            ### for each finalState
            i= 0
            finalStatePosCount = 0

            for items in paretoOfFirstPos:
                steps = 0
                logged = False
                ##reset to state(0,0)
                curPos = (0, 0)

                startPos = curPos

                targetTuple = items
                print ("find target:",i,":",targetTuple)
                targetAction = targetTuple[1]
                # print("action:",targetAction)
                targetKey = (curPos, targetAction)
                calcTarget = targetTuple[0]
                target = getOrignalPareto(calcTarget, Q[targetKey].pareto,  Q[targetKey].R)

                #
                #select Q
                while curPos not in RewardPos:
                    # use the found action above where move curPos forward or in non-finalState to take action
                    takeAction(targetAction)
                    steps += 1
                    if curPos in RewardPos:
                        finalStatePosCount += 1
                        #if looped all Pareto
                        if finalStatePosCount == len(paretoOfFirstPos):
                            finalStateCount += 1

                        if isLogConditionMeet(ceil(finalStateCount + finalStatePosCount/len(paretoOfFirstPos))):
                            logged = False


                        ##add trace logging
                        ##['TrailNumber', 'Timestamp', 'OpenState', 'FinalState', 'RewardPostions','FinalStateReward', 'steps', 'path'])
                        if not logged:
                            log([trailCount, finalStateCount, startPos, curPos, Q[targetKey].pareto, getFinalStateReward(curPos),steps, ''])

                        print("-----------Reach Reward Pos: ",curPos, "\twith action count:", actions, "\tQ:",Q[targetKey].pareto, "\tR", Q[targetKey].R,"\tfinalStateCount:", finalStateCount)

                        curPos = (0, 0)
                        actions = 0
                        steps = 0
                        #target, targetAction, length = Resetinturn(curPos,i)
                        i+=1
                        break ##get next item in paretoinfirstPOS
                    else:
                        ap = []
                        for action in NUM_ACTIONS:
                            key = (curPos,action)
                            a, paretos = getDirectionPareto(curPos, action)
                            #print("Action:", key, "\tparetos:", paretos)
                            ap.append((paretos, Q[key].R, action))
                        apdata = flatDirectionPareto(ap)
                        diff, targetAction, target = getMinP(apdata, target)
                        #print("Found! diff:",diff, "\tTarget Action:", targetAction, "\tnew target value:", target )
                        if actions > 30:
                            print("loooooooooooooop,oooooops")
                            actions=0
                            break
                        if not isLogConditionMeet(finalStateCount) :
                            logged = True

                    print("###From Pos:", curPos, "\t take action:", targetAction, "\ttarget:", target)
                    # takeAction(targetAction)#use the found action above where move curPos forward
                    # steps += 1
        trailCount += 1

    print('################## end of experiment ################')

def getAction(curPos):
    # calculate Q
    currentV=[]
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto=[]
        # print("Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
        for p in Q[currKey].pareto: #数组是值引用，计算时不会更新原值
            p = Q[currKey].R + GAMMA * p #todo: change key
            temppareto.append(p)
        currentV.append((temppareto, action))
    #pareto Q
    #print ("currentV",currentV)
    sortedParento = paretoQ(curPos, currentV) # currentParento = ([[0,1],[1,0]], action)
    #print ("sortedParento", sortedParento)

    # select Q
    targetTuple = random.choice(sortedParento)
    targetAction = targetTuple[1]
    # print("action:",targetAction)
    # targetKey = (curPos, targetAction)
    # calcTarget = targetTuple[0]
    # target = getOrignalPareto(calcTarget, Q[targetKey].pareto,  Q[targetKey].R)

    # print("target:", target)
    return targetAction

def getFinalStateReward(finalState):
    if finalState not in RewardPos:
        raise Exception('##This is not a valid final state!')
    return RewardGrid[finalState[0]][finalState[1]]

def noTrace():
    global curPos, prevAction, prevState, actions
    FinalCount = 0
    curPos = (0, 0)

    #loop
    while FinalCount < 100:
        Action=getAction(curPos)
        takeAction(Action)
        #actions += 1
        if curPos in RewardPos:
            FinalCount += 1
            print("-----------Reach Reward Pos: ",curPos, "\twith action count:", actions, "\tFinalCount:", FinalCount)
            curPos = (0, 0)
            actions = 0
            Action=getAction(curPos)

        print("###From Pos:", curPos, "\t take action:", Action, "\t actions Count:", actions)


def isLogConditionMeet(finalStateCount):
    return (finalStateCount % runSettings['resultInterval'] == 0)           \
           or (runSettings['logLowerFinalState'] and ((finalStateCount < 20 and finalStateCount % 2 == 0)          \
           or (finalStateCount < 100 and finalStateCount % 10 == 0)))

def initializeLogger():

    logFolder = runSettings['logFolder']
    if not os.path.isdir(logFolder):
        os.makedirs(logFolder)
    print('data.{0}.csv'.format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S')))
    FORMAT = '%(message)s'
    logging.basicConfig(format = FORMAT,
                        #filename = runSettings['logFolder'] + 'data%s.csv'.format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S')),
                        handlers = [
                            logging.FileHandler("{0}/{1}".format(runSettings['logFolder'], 'data%s.csv'.format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S')))),
                            logging.StreamHandler()],
                        level = logging.DEBUG)


def log(msg):
    if(isinstance(msg,list) ):
        msg = "|".join(map(str,msg))
    logging.debug(msg)
def initialize():

    global runSettings, logger, logFile
    runSettings = { "totalTrailCount": 1,
                    "finalStateUpperBound": 2,
                    "resultInterval" : 50,
                    "logLowerFinalState" : True,
                    "logFolder" : "./data/log/"
                  }
    initializeLogger()
    log(['TrailNumber', 'Timestamp', 'OpenState', 'FinalState', 'RewardPostions','FinalStateReward', 'steps', 'hyperVolumn','path'])

def cleanUp():
    logFile.close()

def run():
    ##initialize settings
    initialize()

    training1()
    #testing1()
    testinginturn()
    #notrace()
    #cleanUp()

run()
