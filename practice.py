# __author__ = 'X'
#
# import numpy as np
#
# a = np.array([0, 1])
# b = np.array([1, 0])
# c = np.array([1, 3])
# print ("a + b", a + b)
#
# if np.less(a, c).all():
#     print('Yay!!!!! All A < B')
# else:
#     print('Damn! Not all A < B')
#
#
# if np.less(a, b).all():
#     print('Yay!!!!! All A < B')
# else:
#     print('Damn! Not all A < B')

# data1 = [([[0,1],[1,0]],'a'), ([[0,2],[2,0]],'b')]
# data2 = [([0,1],'a'), ([0,2],'b')]
#
#
# def populate(data):
#     """
#         array of actions
#         , which each action is a 2d array,
#         this function turns array of 2d array to simple array of tuple([-1,1], action]
#     :param data: array of tuple(2d array[[-1,2], [0, 3]], action)
#     :return:
#     """
#     dt = []
#     if isinstance(data[0][0][0], list):
#         for item in data:
#             for p in item[0]:
#                 dt.append((p,item[1]))
#         return dt
#     if isinstance(data[0][0][0], int):
#         return data
#
# a = populate(data1)
# print (a)
# b = populate(data2)
# print (b)



# from hv import HyperVolume
# referencePoint = [2, 2, 2]
# hyperVolume = HyperVolume(referencePoint)
# front = [[1, 0, 1], [0, 1, 0]]
# result = hyperVolume.compute(front)

import  numpy as np

# def tickoff00(currParentoafter):
#     nonzero=[]
#     for item in currParentoafter:
#         if not np.array_equal(item, np.array([0, 0])):
#             nonzero.append(item)
#     return nonzero
#
# data=[np.array([0, 0]), np.array([0, 0]), np.array([ -0.2,  24. ]), np.array([  -1.2,  144. ]), np.array([0, 0])]
#
# sure=tickoff00(data)
# print("nonzero:", sure)
i='10'
u='5'
b=(10 - 5)**2
a=((i - u)**2).sum()

print ("b:",b)