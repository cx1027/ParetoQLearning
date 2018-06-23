__author__ = 'X'

# import matplotlib.pyplot as plt
# data = [{"pareto": [{"a":-1,"b":2},  {"a":2, "b":1}],        "action":     'l'}
#         ,{"pareto": [{"a":0,"b":0},  {"a":1, "b":2}],        "action":     'm'}
#          ]

class Pareto(object):

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def readdata(self):
        print ("a:", self.a, "b:", self.b)

# a = Pareto(1,2)
# b = Pareto(3,4)
# c = Pareto(5,6)
# a.readdata()
# b.readdata()
# c.readdata()

items = []
for i in range(5):
    items.append(Pareto(1,2))
for item in items:
    item.readdata()










""""get min value of distance"""""

def getmin(data):
    return min(data, key = lambda item: item[0])

def dev(data, target):
    calc = []
    for item in data:
        calc.append((((item[0] - target)**2).sum(), item[1], item[0]))
    return calc

import  numpy as np

data = np.array([1,1])
moreData = [(np.array([1,2.4]), "A"), (np.array([1 ,2.2]), "B"), (np.array([1,2.4]), "A"), (np.array([1.3,2.2]), "C") ]
target = [1, 2]
GAMA = np.array([3,3])


print(((data-GAMA)**2).sum() )

a = getmin(dev(moreData, target))
print(a)