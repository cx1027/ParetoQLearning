__author__ = 'X'

import matplotlib.pyplot as plt
import numpy as np
# data = [([[-1,2],  [2, 1]],             'a')
#         ,([[0,0],  [1, 2]],             'b')
#         ,([[-1,4], [3, 2]],             'c')
#         ,([[0,0],  [1, -1], [-1, 3 ]],  'd')
#         ,([[-2,4], [5, -2 ]],           'e')
#          ]

# data = [([[0,0],  [0, 0]],             'a')
#         ,([[0,0],  [-1, 2]],             'b')
#         ,([[0,0], [0, 0]],             'c')
#          ]

# data = [([[-1,2],  [2, -1]],             'a')
#         ,([[0,0]],             'b')
#         ,([[-1,4], [3, 2]],             'c')
#         # ,([[0,0],  [1, -1], [-1, 3 ]],  'd')
#         # ,([[-2,4], [5, -2 ]],           'e')
#          ]


# data=[
# ([-1.,  5.], 'a'),
# ([ -2.,  10.], 'a'),
# ([ -3.,  80.], 'a'),
# ([  -4.,  160.], 'a'),
# ([  -6.,  240.], 'a'),
# ([  -8.,  280.], 'a'),
# ([  -9.,  290.], 'a'),
# ([ -10.,  300.], 'a'),
# ([ -14.,  326.], 'a'),
# ([ -15.,  332.], 'a'),
# ([ -17.,  473.], 'a'),
# ([ -18.,  946.], 'a'),
# ([  -19.,  1750.], 'a'),
# ([  -20.,  3500.], 'a'),
# ([  -25.,  3500.], 'a'),
# ([  -20.,  0.], 'a'),
# ([ -20.,  946.], 'a')]

#data=[((-3, 80),'a') ,((-2, 10),'a'),((-1, 5),'a'),((-10, 80),'a'),( (-5, 10),'a'),( (-3, 5),'a')]



data=[
([-6.0, 1.2391642371234208, ],'A'),
([-3.0, 0.1391642371234208, ],'A'),
([-7.0, 1.5191642371234206, ],'A'),
([-2.0, 0.05247813411078717, ],'A'),
([-3.0, 1.0174927113702623, ],'A'),

]
# ([[-1.,  5.]], 'a' ),
# ([[ -2.,  10.]], 'a'),
# ([[ -3.,  80.]], 'b' ),
# ([[  -4.,  160.]], 'b' ),
# ([[  -6.,  240.]], 'b' ),
# ([[  -8.,  280.]], 'b' ),
# ([[  -9.,  290.]], 'b' ),
# ([[ -10.,  300.]], 'b' ),
# ([[ -10.,  300.]], 'b' ),
# ([[ -10.,  300.]], 'b' ),
# ([[ -13.9462932 ,  317.52365672]], 'b' ),
# ([[ -13.94734412,  317.68985953]], 'b' ),
# ([[ -14.94275291,  328.63545529]], 'b' ),
# ([[ -14.97938774,  328.67503816]], 'b' ),
# ([[ -17.43003978,  499.23359448]], 'b' ),
# ([[  -18.43003978,  2249.23359448]],'c')

# def populate(data):
#     """
#         array of actions
#         , which each action is a 2d array,
#         this function turns array of 2d array to simple array of tuple([-1,1], action]
#     :param data: array of tuple(2d array[[-1,2], [0, 3]], action)
#     :return:
#     """
#     dt = []
#     for item in data:
#         for p in item[0]:
#             dt.append((p,item[1]))
#     return dt

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
    if isinstance(data[0][0][0], int) or isinstance(data[0][0][0], float):
        return data


# def paretoArray(paretoInAction):
#     """
#         pareto all values with action
#     :param paretoInAction: array of tuple(array[-1,2], action)
#     :return: pareto with action array of tuple([-1,2], action)
#     """
#     data = populate(paretoInAction)
#     print("source:", data)
#     sortedBy1st = sorted(data, key = lambda x: x[0][0], reverse = True) #sort tuple with first value
#     print("sorted:",sortedBy1st)
#
#     pareto = []
#     pareto.append(sortedBy1st[0])
#     for item in sortedBy1st[1:]:
#         if item[0][1] >= pareto[-1][0][1]: #get the last item to compare the 2nd value in array
#             if item[0][0] == pareto[-1][0][0] and item[0][1] == pareto[-1][0][1]:
#                 if len(pareto) > 1:
#                     pareto.pop(-1) #both equal
#             if item[0][1] > pareto[-1][0][1]:
#                 pareto.append(item)
#     print("pareto:", pareto)
#     return pareto
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
                if item[0][0] == pareto[-1][0][0] :
                    if item[0][1] > pareto[-1][0][1]:
                        pareto.pop(-1)
                        pareto.append(item)
                #item[right]< p[right]
                elif item[0][1] > pareto[-1][0][1]:
                    pareto.append(item)
        # print("pareto:", pareto)
    return pareto



#testing
def run():
    Xsp = []
    Ysp = []
    Zsp = []
    dp = populate(data)
    pdata = paretoArray(dp)
    print("paretodata",pdata)
    print ("size",pdata.__sizeof__())


    Xs = []
    Ys = []
    Zs = []

    for p in dp:
        item, z = p
        Zs.append(z)
        Xs.append(item[0])
        Ys.append(item[1])
    plt.scatter(Xs, Ys)

    for p in pdata:
        item, z = p
        Zsp.append(z)
        Xsp.append(item[0])
        Ysp.append(item[1])

    # Then plot the Pareto frontier on top
    plt.plot(Xsp, Ysp)

    for i, txt in enumerate(Zsp):
        plt.annotate(txt, (Xsp[i],Ysp[i]),   textcoords='offset points')
    plt.show()

run()