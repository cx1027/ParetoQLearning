# -*- coding: utf-8 -*-
__author__ = 'X'
# update Pareto
# update testing part

import numpy as np
import sys, random
from enum import Enum
import os, csv, logging
import mpmath
import datetime


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
    ACTION_EAST = 2,
    ACTION_SOUTH = 3,
    ACTION_WEST = 4


DEBUG = False


class QAction(object):
    action = None  # action is from NUM_ACTIONS
    pareto = []  # store (1,99),(2,98) etc.
    R = None

    def __init__(self, rInit=None, qInit=None, direction=None):
        self.action = direction
        # self.pareto = np.array[(0,0)] 'except finalState'
        self.pareto = []
        self.pareto.append(qInit)
        self.R = rInit
        self.N = N_INIT

    def __str__(self):
        return 'Action:%s \t R:%s \t N:%s \t Pareto:%s' % (self.action, self.R, self.N, self.pareto)


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


# learning tables
Q = {}
trash = {}

Config = {}

GRID_x = 21
GRID_y = 14
Q_INIT = np.array([0, 0])
R_INIT = np.array([0, 0])
N_INIT = 0
FINAL_POSITION_REWARD = 1750
REF_POINT = np.array([-10, -10])

# learning parameters
GAMMA = 1

FinalCount = 0

# globals
grid = []
curPos = (0, 0)
actions = 0

# remembered information
prevState = None
prevAction = None
prevReward = None

# read grid from file
# RewardGrid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [5, 80, 0, 0, 0, 0, 0, 0, 0, 0],
#               [5, 5, 120, 0, 0, 0, 0, 0, 0, 0],
#               [5, 5, 5, 140, 145, 150, 0, 0, 0, 0],
#               [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
#               [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
#               [5, 5, 5, 5, 5, 5, 163, 166, 0, 0],
#               [5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
#               [5, 5, 5, 5, 5, 5, 5, 5, 473, 0],
#               [5, 5, 5, 5, 5, 5, 5, 5, 5, 1750]
#               ] 10X11

#15X22
RewardGrid =[[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
[5,0,0,0,0,30,5,5,5,5,5,5,5,5,5],
[5,0,5,0,0,0,5,5,5,5,5,5,5,5,5],
[5,0,5,0,0,0,5,5,5,5,5,5,5,5,5],
[5,20,5,0,0,0,5,5,5,5,5,5,5,5,5],
[5,5,5,5,0,0,5,5,5,5,5,5,5,5,5],
[5,5,5,5,0,0,5,5,5,5,5,5,5,5,5],
[5,5,5,5,5,0,5,5,5,5,5,5,5,5,5],
[5,5,5,5,5,0,0,0,0,30,5,5,5,5,5],
[5,5,5,5,5,0,5,0,0,0,5,5,5,5,5],
[5,5,5,5,5,0,5,0,0,0,5,5,5,5,5],
[5,5,5,5,5,20,5,0,0,0,5,5,5,5,5],
[5,5,5,5,5,5,5,5,0,0,5,5,5,5,5],
[5,5,5,5,5,5,5,5,0,0,5,5,5,5,5],
[5,5,5,5,5,5,5,5,5,0,5,5,5,5,5],
[5,5,5,5,5,5,5,5,5,0,0,0,0,30,5],
[5,5,5,5,5,5,5,5,5,0,5,0,0,0,5],
[5,5,5,5,5,5,5,5,5,0,5,0,0,0,5],
[5,5,5,5,5,5,5,5,5,20,5,0,0,0,5],
[5,5,5,5,5,5,5,5,5,5,5,5,0,0,5],
[5,5,5,5,5,5,5,5,5,5,5,5,0,0,5],
[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]]

OpenPos = (
           (1, 1), (1, 2), (1, 3), (1, 4),
           (2, 1), (2, 3), (2, 4), (2, 5),
           (3, 1), (3, 3), (3, 5),
           (4, 3), (4, 5),
           (5, 5), (6, 5),(7, 5),

           (8, 5), (8, 6), (8, 7), (8, 8),
           (9, 5), (9, 7), (9, 8), (9, 9),
           (10, 5), (10, 7), (10, 8),
           (11, 7), (11, 9),
           (12, 9), (13, 9),(14, 9),

           (15, 9), (15, 10), (15, 11), (15, 12),
           (16, 9), (16, 11), (16, 12), (16, 13),
           (17, 9), (17, 11), (17, 13),
           (18, 11), (18, 13),
           (19, 13), (20, 13)

           )

RewardPos = [(4, 1), (1, 5), (11, 5), (8, 9), (18, 9), (15, 13)]


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
    # mark previous State
    prevState = curPos
    prevAction = action

    reward = np.array([-1, 0])
    try:
        if action == NUM_ACTIONS.ACTION_EAST:
            if y < GRID_y:
                y += 1
        elif action == NUM_ACTIONS.ACTION_NORTH:
            if x > 0:
                x -= 1
        elif action == NUM_ACTIONS.ACTION_WEST:
            if y > 0 and curPos != (5, 6) and curPos != (6, 6) and curPos != (8, 8):
                y -= 1
        elif action == NUM_ACTIONS.ACTION_SOUTH:
            if x < GRID_x:
                x += 1
        # if 到边界，there is no reward??????!!!!!
        curPos = (x, y)
        # if curPos == (0, 0) and trashes == 0 and reward != REWARD_PICKUP:
        if curPos in RewardPos:
            reward = np.array([-1, RewardGrid[x][y]])

        # Set N:
        actions += 1
    except Exception as e:
        print(e)
        print('x,y:', x, y)
    return reward


def getInitialValuesByPosition(curPos):
    x, y = curPos
    if curPos in RewardPos:
        return (np.array([-1, RewardGrid[x][y]]), N_INIT)
    else:
        return (R_INIT, Q_INIT, N_INIT)


def paretoArray(data):
    """
        pareto all values with action
    :param paretoInAction: array of tuple(array[-1,2], action)
    :return: pareto with action array of tuple([-1,2], action)
    """
    # data = populate(paretoInAction)
    # tuple=[([-1,2], action),([-1,2], action)]
    if isinstance(data[0], tuple):
        # print("source:", data)
        sortedBy1st = sorted(data, key=lambda x: x[0][0], reverse=True)  # sort tuple with first value
        # print("sorted:",sortedBy1st)

        pareto = []
        pareto.append(sortedBy1st[0])
        for item in sortedBy1st[1:]:
            if item[0][1] >= pareto[-1][0][1]:  # get the last item to compare the 2nd value in array
                # item[left]= p[left]
                if item[0][0] == pareto[-1][0][0]:
                    if item[0][1] > pareto[-1][0][1]:
                        pareto.pop(-1)
                        pareto.append(item)
                # item[right]< p[right]
                elif item[0][1] > pareto[-1][0][1]:
                    pareto.append(item)
        # print("pareto:", pareto)
        return pareto
    # ndarray=[[-1,2], [1,2]]
    if isinstance(data[0], np.ndarray):
        # print("source:", data)
        sortedBy1st = sorted(data, key=lambda x: x[0], reverse=True)  # sort tuple with first value
        # print("sorted:",sortedBy1st)

        pareto = []
        pareto.append(sortedBy1st[0])
        for item in sortedBy1st[1:]:
            if item[1] >= pareto[-1][1]:  # get the last item to compare the 2nd value in array
                # if item[0] == pareto[-1][0] and item[1] == pareto[-1][1]:
                #     if len(pareto) > 1:
                #         pareto.pop(-1) #both equal
                # if item[1] > pareto[-1][1]:
                #     pareto.append(item)

                # item[left]= p[left]
                if item[0] == pareto[-1][0]:
                    if item[1] > pareto[-1][1]:
                        pareto.pop(-1)
                        pareto.append(item)
                # item[right]< p[right]
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
                dt.append((p, item[1]))
        return dt
    if isinstance(data[0][0][0], int):
        return data


def flatDirectionPareto(data):
    flat = []
    try:
        if not (isinstance(data[0], tuple) and isinstance(data[0][0], list) and isinstance(data[0][0][0], np.ndarray)):
            raise ("!!! Expecting data in format: [([[2,1],[1,2]], action, R),(...)].")
        for item in data:
            for ditem in item[0]:
                flat.append((ditem, item[1], item[2]))
    except Exception as ex:
        print(ex)
    return flat


# returns the paretp Q-value for an action in the specified state
def paretoQ(curPos, currentParento=[]):
    allValues = []
    pareto = []
    # has value in currentParento
    if len(currentParento) != 0:
        allValues = currentParento
    # no value in currentParento
    else:
        for curAction in NUM_ACTIONS:
            key = (curPos, curAction)
            # print('paretoQ--------CurPos:',curPos, ' Action:', curAction)
            if not key in Q:
                (ri, qi, ni) = getInitialValuesByPosition(curPos)
                Q[key] = QAction(ri, qi, curAction)
            # get pareto from each direction in  Q[key] (QClass)
            if len(Q[key].pareto) != 0:
                allValues.extend((Q[key].pareto, curAction))
    # pareto sorting
    return paretoArray(populate(allValues))


def getPareto(curPos):
    pareto = []
    if curPos not in RewardPos:
        for curAction in NUM_ACTIONS:
            key = (curPos, curAction)
            if key in Q:
                # get pareto from each direction in  Q[key] (QClass)
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
        pareto.extend(Q[key].pareto)

        # if key in Q:
        #     # get pareto from each direction in  Q[key] (QClass)
        #     pareto.extend(Q[key].pareto)

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
    directionParentoCounts = {}  # store count for each direction

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
        if (bool(directionParentoCounts)):
            bestAction = max(directionParentoCounts, key=directionParentoCounts.get)
        else:
            bestAction = random.choice(list(NUM_ACTIONS))

    return bestAction


def tickoff00(currParentoafter):
    nonzero = []
    for item in currParentoafter:
        if not np.array_equal(item, np.array([0, 0])):
            nonzero.append(item)
    return nonzero


def updateQ(reward, curPos):
    global prevState, prevAction, prevReward, RewardGrid, actions
    # curPos is the first point
    if prevState == None and prevAction == None:
        # Q[key].R = Q[key].R + (reward - Q[key].R) / actions
        return 0

    # 'normal point or last 3 points'
    else:
        # R 'later'
        key = (prevState, prevAction)  #
        (x, y) = curPos

        # Q
        # get Qvalue of current block
        if curPos not in RewardPos:
            # inital 4 square of this block
            for action in NUM_ACTIONS:
                currKey = (curPos, action)
                if not currKey in Q:  # why if not currKey?????
                    (ri, qi, ni) = getInitialValuesByPosition(curPos)
                    Q[currKey] = QAction(ri, qi, action)

            #calc tempQ=R+ temp Q
            # for action in NUM_ACTIONS:
            #     currKey = (curPos, action) #TODO:to be changed by daciba
            #     if not key in Q:
            #         (ri, qi) = getInitialValuesByPosition(curPos)
            #         Q[currKey] = QAction(ri, qi, action)

            currParento = []

            # calc p for each direction of curr Pos
            for action in NUM_ACTIONS:
                currKey = (curPos, action)
                # print ("before", Q[currKey].pareto)
                for p in Q[currKey].pareto:  # value reference, won't affect original value after assigned
                    p = Q[currKey].R + GAMMA * p  # todo: change key
                    currParento.append(p)
                # print ("after", Q[currKey].pareto)
            # get pareto of current state
            # assign to triangle
            if not key in Q:  # why if not currKey?????
                (ri, qi, ni) = getInitialValuesByPosition(prevState)
                Q[key] = QAction(ri, qi, prevAction)
            currParentoafter = paretoQ(curPos, currParento)
            ################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            currParentoafter = tickoff00(currParentoafter)  ###########################################################
            ####################################################################################################
            if len(currParentoafter) >= 1:
                Q[key].pareto = currParentoafter
            # print ('trangle value:', Q[key].pareto)
            # to next state
            prevState = curPos

        if (x, y) in RewardPos:
            # if(curPos==(3,2)):
            #   print("3,2")
            currParento = getPareto(curPos)

            # get pareto of current state
            # assign to triangle
                    # if key not in Q:
                    #     (ri, qi, ni) = getInitialValuesByPosition(prevState)
                    #     Q[key] = QAction(ri, qi, ni)

            currParento.extend(Q[key].pareto)
            Q[key].pareto = paretoQ(curPos, currParento)
            # to next position

            prevState = None
            prevAction = None
            prevReward = None

        # in final state
        # R:calculate R for previous state
        # Q[key].R = prevReward
        # if not key in Q:
        #     print("***********key not in Q:", key)
        #     (ri, qi) = getInitialValuesByPosition(prevState)
        #     Q[key] = QAction(ri, qi, key)

        Q[key].N += 1
        Q[key].R = Q[key].R + (reward - Q[key].R) / Q[key].N
        # print('key:',key, '\t\tR:', Q[key].R, '=\t\tPR:', prevR, '+ \t(r:' ,reward, '- PR:',prevR, ')/  #:', actions)
        prevReward = Q[key].R
        # print ('Reward:', prevReward, ' Actions:', actions)

        if (x, y) in RewardPos:
            actions = 0


def finalParetoprint(curPos):
    # calculate Q
    currentV = []
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto = []
        if currKey in Q:
            print("final Pareto Print: Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
            for p in Q[currKey].pareto:  # assign by value not reference
                p = Q[currKey].R + GAMMA * p  # todo: change key
                temppareto.append(p)
        currentV.append((temppareto, action))
    # pareto Q
    print("currentV", currentV)
    sortedParento = paretoQ(curPos, currentV)  # currentParento = ([[0,1],[1,0]], action)
    print("sortedParento", sortedParento)


def DisplayFinal():
    for Pos in OpenPos:
        print("CurPos:", Pos)
        finalParetoprint(Pos)


def training1():
    global curPos, RewardGrid, actions, GAMMA, N, FinalCount

    trailCount = 0

    printGrid(RewardGrid)

    ALPHA = 1.0
    GAMMA = 1

    #print("ALPHA: ", ALPHA, "\nGAMMA: ", GAMMA, "\nGRID_LENGTH: ", GRID_y, "\nGRID_WIDE: ", GRID_x, )
    print("ALPHA: ", ALPHA, "\nGAMMA: ", GAMMA, "\nGRID_LENGTH: ", len(RewardGrid), "\nGRID_WIDE: ", len(RewardGrid[0]), )

    # total new start
    # curPos = random.choice(GridState)
    # action = random.choice(list(NUM_ACTIONS)) #reward,state,avgreward
    curPos = (0, 0)
    ### trail
    while trailCount < runSettings['totalTrailCount']:
        ### finalStateUpperBound
        finalStateCount = 0
        while FinalCount <= runSettings['finalStateUpperBound']:

            # if curPos ==(9,9):
            #     print (actions)
            # if FinalCount == 0:
            #     action = NUM_ACTIONS.ACTION_SOUTH
            # else:
            # action = selectBestAction(curPos)
            action = random.choice(list(NUM_ACTIONS))
            # print('Cur Pos:', curPos, ' FinalCount:', FinalCount, ' Action:',action)

            reward = takeAction(action)
            updateQ(reward, curPos)

            # print('*** finalCount:', FinalCount)
            if curPos in RewardPos:
                FinalCount += 1
                # print("final curPos",curPos)
                # DisplayGrid()
                finalStateCount += 1
                if isLogConditionMeet(finalStateCount, 0):
                    #testingInTurn()
                    positions = OpenPos
                    #positions = [(0, 0)]

                    hyperSum = 0
                    for pos in OpenPos:
                        hyperSum += calc_position_hyper_volume(pos)
                    for pos in positions:
                        runTrace(pos, trailCount, finalStateCount, hyperSum)

                # after testing
                #curPos = (0, 0)
                curPos = random.choice(OpenPos)
                # print ('ramdon pos:', curPos)

        trailCount += 1
        FinalCount = 0

    # DisplayGrid()
    # DisplayFinal()


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
        #TODO:equal doesnt work
        if np.equal(caculateQ, R + GAMMA * q).all():
            targetQ = q
            break
    # if targetQ == None:
    #     raise("Couldn't find orginal Q")
    return targetQ


def Reset(curPos):
    # calculate Q
    currentV = []
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto = []
        # print("Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
        for p in Q[currKey].pareto:  #assign by value but not reference
            p = Q[currKey].R + GAMMA * p  # todo: change key
            temppareto.append(p)
        currentV.append((temppareto, action))
    # pareto Q
    print("currentV", currentV)
    sortedParento = paretoQ(curPos, currentV)  # currentParento = ([[0,1],[1,0]], action)
    print("sortedParento", sortedParento)

    # select Q
    targetTuple = random.choice(sortedParento)
    targetAction = targetTuple[1]
    # print("action:",targetAction)
    targetKey = (curPos, targetAction)
    calcTarget = targetTuple[0]
    target = getOrignalPareto(calcTarget, Q[targetKey].pareto, Q[targetKey].R)

    # print("target:", target)
    return target, targetAction


def Resetinturn(curPos):
    '''

    :param curPos:
    :return:
    '''
    # calculate Q
    currentV = []
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto = []
        print("Cur Pos", currKey)
                # if currKey not in Q:
                #     (ri, qi, ni) = getInitialValuesByPosition(curPos)
                #     Q[currKey] = QAction(ri, qi, action)
        for p in Q[currKey].pareto:  #assign by value but not reference
            p = Q[currKey].R + GAMMA * p  # todo: change key
            temppareto.append(p)
        currentV.append((temppareto, action))
    # pareto Q
    print("currentV", currentV)
    sortedParento = paretoQ(curPos, currentV)  # currentParento = ([[0,1],[1,0]], action)
    length = len(sortedParento)
    print("sortedParento", sortedParento)

    return sortedParento

    # select Q
    # targetTuple = sortedParento
    # targetAction = targetTuple[1]
    # # print("action:",targetAction)
    # targetKey = (curPos, targetAction)
    # calcTarget = targetTuple[0]
    # target = getOrignalPareto(calcTarget, Q[targetKey].pareto,  Q[targetKey].R)
    #
    # # print("target:", target)
    # return target, targetAction,length


def getMinP(paretos, target):
    """pareto = (np.array([1,1]),Action, R)"""
    calcP = []
    for p in paretos:
        calc = GAMMA * p[0] + p[1]
        calcP.append((((calc - target) ** 2).sum(), p[2], p[0]))

    return min(calcP, key=lambda item: item[0])


def testing1():
    global curPos, prevAction, prevState, actions
    FinalCount = 0
    curPos = (0, 0)

    target, targetAction = Reset(curPos)
    print("target:", target)
    takeAction(targetAction)

    # loop
    while FinalCount < 1200:
        if curPos in RewardPos:
            FinalCount += 1
            print("-----------Reach Reward Pos: ", curPos, "\twith action count:", actions, "\tFinalCount:", FinalCount)
            curPos = (0, 0)
            actions = 0
            target, targetAction = Reset(curPos)
            print("target:", target)

        else:
            ap = []
            for action in NUM_ACTIONS:
                key = (curPos, action)
                a, paretos = getDirectionPareto(curPos, action)
                # print("Action:", key, "\tparetos:", paretos)
                ap.append((paretos, Q[key].R, action))
            apdata = flatDirectionPareto(ap)
            diff, targetAction, target = getMinP(apdata, target)
            # print("Found! diff:",diff, "\tTarget Action:", targetAction, "\tnew target value:", target )

        #print("###From Pos:", curPos, "\t take action:", targetAction, "\ttarget:", target)
        takeAction(targetAction)  # use the found action above where move curPos forward

    # print("Key:", key)


def testingInTurn():
    global curPos, prevAction, prevState, actions, FinalCount
    trailCount = 0

    paretoOfFirstPos = Resetinturn((0, 0))
    print("pareto in 0,0:", paretoOfFirstPos, "\n number:", len(paretoOfFirstPos))

    ### trail
    while trailCount < runSettings['totalTrailCount']:
        ### finalStateUpperBound
        finalStateCount = 0
        while finalStateCount <= runSettings['finalStateUpperBound']:
            ### for each finalState
            i = 0
            finalStatePosCount = 0

            for items in paretoOfFirstPos:
                steps = 0
                path = []
                logged = True
                ##reset to state(0,0)
                curPos = (0, 0)

                startPos = curPos

                targetTuple = items
                print("find target:", i, ":", targetTuple)
                targetAction = targetTuple[1]
                # print("action:",targetAction)
                targetKey = (curPos, targetAction)
                calcTarget = targetTuple[0]
                target = getOrignalPareto(calcTarget, Q[targetKey].pareto, Q[targetKey].R)

                #
                # select Q
                while curPos not in RewardPos:
                    # use the found action above where move curPos forward or in non-finalState to take action
                    takeAction(targetAction)
                    steps += 1
                    path.append(curPos)
                    if curPos in RewardPos:
                        finalStatePosCount += 1

                        if isLogConditionMeet(mpmath.ceil(finalStateCount + finalStatePosCount / len(paretoOfFirstPos))):
                            logged = False

                        ##add trace logging
                        ##['TrailNumber', 'Timestamp', 'OpenState', 'FinalState', 'RewardPostions','FinalStateReward', 'steps', 'path'])
                        if not logged:
                            posReward = getFinalStateReward(curPos)
                            log([trailCount, finalStateCount, startPos, curPos, posReward, steps
                                    , calc_position_hyper_volume(curPos)
                                    , checkRewardInGrid(Q[targetKey].pareto, posReward)
                                    , Q[targetKey].pareto])

                        print("-----------Reach Reward Pos: ", curPos, "\twith action count:", actions, "\tQ:",
                              Q[targetKey].pareto, "\tR", Q[targetKey].R, "\tfinalStateCount:", finalStateCount)

                        if finalStatePosCount == len(paretoOfFirstPos):
                            finalStateCount += 1
                        curPos = (0, 0)
                        actions = 0
                        steps = 0
                        # target, targetAction, length = Resetinturn(curPos,i)
                        i += 1
                        break  ##get next item in paretoinfirstPOS
                    else:
                        ap = []
                        for action in NUM_ACTIONS:
                            key = (curPos, action)
                            a, paretos = getDirectionPareto(curPos, action)
                            # print("Action:", key, "\tparetos:", paretos)
                            ap.append((paretos, Q[key].R, action))
                        apdata = flatDirectionPareto(ap)
                        diff, targetAction, target = getMinP(apdata, target)
                        # print("Found! diff:",diff, "\tTarget Action:", targetAction, "\tnew target value:", target )
                        if actions > 30:
                            print("loooooooooooooop,oooooops")
                            actions = 0
                            break
                        if not isLogConditionMeet(mpmath.ceil(finalStateCount + finalStatePosCount / len(paretoOfFirstPos))):
                            logged = True

                    print("###From Pos:", curPos, "\t take action:", targetAction, "\ttarget:", target)
                    # takeAction(targetAction)#use the found action above where move curPos forward
                    # steps += 1
        trailCount += 1

    print('################## end of experiment ################')


def getAction(curPos):
    # calculate Q
    currentV = []
    for action in NUM_ACTIONS:
        currKey = (curPos, action)
        temppareto = []
        # print("Cur Pos", currKey, "\tPareto:", Q[currKey].pareto)
        for p in Q[currKey].pareto:  #assign by value but not reference
            p = Q[currKey].R + GAMMA * p  # todo: change key
            temppareto.append(p)
        currentV.append((temppareto, action))
    # pareto Q
    # print ("currentV",currentV)
    sortedParento = paretoQ(curPos, currentV)  # currentParento = ([[0,1],[1,0]], action)
    # print ("sortedParento", sortedParento)

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


def checkRewardInGrid(paretos, reward):
    ret = list(filter(lambda x: x[1] == reward, paretos))
    if len(ret) > 0:
        return 1
    else:
        return 0


def noTrace():
    global curPos, prevAction, prevState, actions
    FinalCount = 0
    curPos = (0, 0)

    # loop
    while FinalCount < 100:
        action = getAction(curPos)
        takeAction(action)
        # actions += 1
        if curPos in RewardPos:
            FinalCount += 1
            print("-----------Reach Reward Pos: ", curPos, "\twith action count:", actions, "\tFinalCount:", FinalCount)
            curPos = (0, 0)
            actions = 0
            action = getAction(curPos)

        print("###From Pos:", curPos, "\t take action:", action, "\t actions Count:", actions)


def runTrace(position, trailCount, finalStateCount, hyperVol):
    global curPos
    ### for each finalState
    i = 0
    finalStatePosCount = 0

    paretoOfFirstPos = Resetinturn(position)
    print("pareto in 0,0:", paretoOfFirstPos, "\n number:", len(paretoOfFirstPos))

    for items in paretoOfFirstPos:
        steps = 0
        path = []
        logged = True
        ##reset to state(0,0)
        # curPos = (0, 0)
        curPos = position

        startPos = curPos
        path.append(curPos)

        targetTuple = items
        print("find target:", i, ":", targetTuple)
        targetAction = targetTuple[1]
        # print("action:",targetAction)
        targetKey = (curPos, targetAction)
        calcTarget = targetTuple[0]
        target = getOrignalPareto(calcTarget, Q[targetKey].pareto, Q[targetKey].R)
        #
        # select Q
        while curPos not in RewardPos:
            # use the found action above where move curPos forward or in non-finalState to take action
            takeAction(targetAction)
            steps += 1
            path.append(curPos)
            # print(trailCount, finalStateCount, startPos, curPos)
            if curPos in RewardPos:
                finalStatePosCount += 1

                if isLogConditionMeet(finalStateCount, finalStatePosCount / len(paretoOfFirstPos)):
                    logged = False

                ##add trace logging
                if not logged:
                    posReward = getFinalStateReward(curPos)
                    log([trailCount, finalStateCount, startPos, curPos, posReward
                            , steps, hyperVol
                            , checkRewardInGrid(Q[targetKey].pareto, posReward)
                            , checkRewardInGrid(Q[targetKey].pareto, FINAL_POSITION_REWARD)
                            , Q[targetKey].pareto
                            , path])

                print("-----------Reach Reward Pos: ", curPos, "\twith action count:", actions, "\tQ:",
                      Q[targetKey].pareto, "\tR", Q[targetKey].R, "\tfinalStateCount:", finalStateCount)

                # if finalStatePosCount == len(paretoOfFirstPos):
                #     finalStateCount += 1
                curPos = (0, 0)
                steps = 0
                # target, targetAction, length = Resetinturn(curPos,i)
                i += 1
                break  ##get next item in paretoinfirstPOS
            else:
                ap = []
                for action in NUM_ACTIONS:
                    key = (curPos, action)
                    if not key in Q:
                        Q[key] = QAction(R_INIT, Q_INIT, action)
                    act, paretos = getDirectionPareto(curPos, action)
                    #print("Action:", act, "key:", key, "paretos:", paretos)
                    ap.append((paretos, Q[key].R, action))
                apdata = flatDirectionPareto(ap)
                diff, targetAction, target = getMinP(apdata, target)
                # print("Found! diff:",diff, "\tTarget Action:", targetAction, "\tnew target value:", target )

                if not isLogConditionMeet(finalStateCount, finalStatePosCount / len(paretoOfFirstPos)):
                    logged = True

                if steps > 30:
                    print("loooooooooooooop,oooooops")
                    steps = 0
                    posReward = 0  ### failed to reach to final state, set 0 as fake reward
                    # log([trailCount, finalStateCount, startPos, curPos, Q[targetKey].pareto, posReward, steps,
                    #      checkRewardInGrid(Q[targetKey].pareto, posReward), 0], calc_position_hyper_volume(curPos))
                    break

            print("###From Pos:", curPos, "\t take action:", targetAction, "\ttarget:", target)
            # takeAction(targetAction)#use the found action above where move curPos forward
            # steps += 1



def isLogConditionMeet(finalStateCount, turns):
    return (turns <= 1 ) and ( (finalStateCount % runSettings['resultInterval'] == 0) \
           or (runSettings['logLowerFinalState'] and ((finalStateCount < 20 and finalStateCount % 2 == 0) \
                                                      or (finalStateCount < 100 and finalStateCount % 10 == 0))) )


def initializeLogger():
    logFolder = runSettings['logFolder']
    if not os.path.isdir(logFolder):
        os.makedirs(logFolder)
    fileName = 'qlMaze.data.{0}.csv'.format(datetime.datetime.today().strftime('%Y%m%d-%H%M%S'))
    # print(fileName)
    FORMAT = '%(message)s'
    # logging.basicConfig(format=FORMAT,
    #                     handlers=[
    #                         logging.FileHandler("{0}/{1}".format(runSettings['logFolder'], fileName)),
    #                         logging.StreamHandler()],
    #                     level=logging.DEBUG)

    logging.basicConfig(format= FORMAT, #'%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename="{0}/{1}".format(runSettings['logFolder'], fileName),
                        level=logging.DEBUG)
    #logging.debug("This is a debug message")


def log(msg):
    if (isinstance(msg, list)):
        msg = "|".join(map(str, msg))
    logging.debug(msg)


def initialize():
    global runSettings
    runSettings = {  # 'trainingCount': 3500,
        "totalTrailCount": 30,
        "finalStateUpperBound": 3500,
        "resultInterval": 50,
        "logLowerFinalState": True,
        "logFolder": "./data/log/"
    }
    initializeQ()
    initializeLogger()
    # file header
    log(['TrailNumber', 'Timestamp', 'OpenState', 'FinalState', 'RewardPostions', 'FinalStateReward', 'steps', 'hypervol',
         'Matched', 'MatchedFinal', 'paretos', 'path'])

def initializeQ():
    for pos in OpenPos:
        for action in NUM_ACTIONS:
            key = (pos, action)
            if key not in Q:
                (ri, qi, ni) = getInitialValuesByPosition(pos)
                Q[key] = QAction(ri, qi, action)

def calc_position_hyper_volume(curPos):
    tmp_paretos = []
    # key = (curPos, NUM_ACTIONS.ACTION_NORTH)
    # if key not in Q:
    #     Q[key] = QAction(R_INIT, Q_INIT, NUM_ACTIONS.ACTION_NORTH)
    # paretos.extend(Q[key].pareto)
    # key = (curPos, NUM_ACTIONS.ACTION_EAST)
    # paretos.extend(Q[key].pareto)
    # key = (curPos, NUM_ACTIONS.ACTION_SOUTH)
    # paretos.extend(Q[key].pareto)
    # key = (curPos, NUM_ACTIONS.ACTION_WEST)
    # paretos.extend(Q[key].pareto)
    for action in NUM_ACTIONS:
        key = (curPos, action)
        tmp_paretos.extend(Q[key].pareto)
    paretos = paretoArray(tmp_paretos)
    pareto_sorted = sorted(paretos, key=lambda x: x[0], reverse=False)
    return calc_hyper_volume(pareto_sorted, REF_POINT)




def calc_hyper_volume(paretos, ref_point):
    sum = 0
    for idx, item in enumerate(paretos):
        if idx == len(paretos) - 1:
            sum += mpmath.fabs(paretos[idx][0] - ref_point[0]) * mpmath.fabs(paretos[idx][1] - ref_point[1])
        else:
            sum += mpmath.fabs(paretos[idx][0] - paretos[idx + 1][0] ) * mpmath.fabs(paretos[idx][0] - ref_point[1])

    return sum

def run():
    ##initialize settings
    initialize()

    training1()
    # testing1()
    #testingInTurn()
    # notrace()
    # cleanUp()


run()

