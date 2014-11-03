import pandas as pd
import numpy as np
import os
from pandas import read_csv
from pandas import DataFrame as df
import logging
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)
from collections import Counter
import networkx as nx
from pylab import *
import sys
import numpy
import networkx as nx
import re

f = open('data/featureList.txt')
features = f.readlines()
features = map(lambda x: x[:-1], features)

filem = open('data/features.txt')
fm = filem.readlines()
fm = map(lambda x: x[:-1], fm)

fmatrix = df(columns=features)

def extract(fma, user, x):
    match = re.search(r'([\w;]+);(\d+)', x)
    m = match.groups()
    fma.ix[user, m[0]] = m[1]
    
def doline(x):
    l = x.split(' ')
    fma = df(columns=features)
    for i in range(1,len(l)):
		extract(fma, l[0],l[i])
    return fma.values[0]

from multiprocessing import Pool
pool = Pool(4)


logging.info('Started parallel map')
fmatrix = df(pool.map(doline, fm),columns=features)
logging.info('Done!')

fmatrix = fmatrix.astype(float)
fmatrix.to_csv('data/fmatrix.csv',index=False)


fmatrix = read_csv('data/fmatrix.csv')

path = 'data/egonets/'
egonets =os.listdir(path)

friends = []

for fle in egonets:
    f = open(path+fle)
    lines = f.readlines()
    lines = map(lambda x: x[:-1], lines)
    
    #print lines
    #raw_input("A1 Press Enter to continue...")

    for line in lines:
        l2 = line.split(': ')
        for sec in l2[1].split(' '):
    		friends.append((fle[:-7],l2[0],sec))

friends = df(friends, columns=['user','friend','friend_of_friend'])

#forever_alone are people who has no friends but the main user for this egonet
#if you find someone who dont present in this table, count him as separate group (check if it legal by youself)
forever_alone = friends[friends.friend_of_friend == ''].drop(['friend_of_friend'], axis=1).reset_index().drop(['index'], axis=1).astype('float').astype('int')

friends = friends[friends.friend_of_friend != ''].reset_index().drop(['index'], axis=1).astype('float').astype('int')

friends.to_csv('data/friends.csv',index=False)

friends[['friend','friend_of_friend']].rename(columns={'friend':'Source','friend_of_friend':'Target'}).to_csv('data/friends_gephi.csv',index=False)

path = 'data/Training/'
training = os.listdir(path)

circles = []

for fle in training:
    f = open(path+fle)
    lines = f.readlines()
    lines = map(lambda x: x[:-1], lines)
    
    for line in lines:
        l2 = line.split(': ')
        for sec in l2[1].split(' '):
            circles.append((fle[:-8],l2[0][6:],sec))
            
circles = df(circles, columns=['user','circle','friend']).astype('float').astype('int')

circles.to_csv('data/circles.csv',index=False)

merged = pd.merge(friends,circles, how='left')
merged.dtypes

merged.to_csv('data/friend_and_circles.csv', index=False)

users = list(circles.user.unique())
train = list((Counter(map(lambda x: int(x[:-7]),egonets)) & Counter(map(lambda x: int(x[:-8]),training))).elements())
test = list((Counter(map(lambda x: int(x[:-7]),egonets)) - Counter(train)).elements())
path = 'data/egonets/'

