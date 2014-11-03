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
import random
from pylab import *
import sys
import heapq

f = open('data/featureList.txt')
features = f.readlines()
features = map(lambda x: x[:-1], features)

friends_circles = read_csv('data/friend_and_circles.csv')
circles = read_csv('data/circles.csv')
friends = read_csv('data/friends.csv')
fmatrix = read_csv('data/fmatrix.csv')

path = 'data/Training/'
training = os.listdir(path)

path = 'data/egonets/'
egonets = os.listdir(path)

users = list(circles.user.unique())
train = list((Counter(map(lambda x: int(x[:-7]),egonets)) & Counter(map(lambda x: int(x[:-8]),training))).elements())
test = list((Counter(map(lambda x: int(x[:-7]),egonets)) - Counter(train)).elements())

def maxrank(user):
    G = nx.Graph()
    for i,j in friends[friends.user == user].iterrows():
        G.add_edge(j.values[1],j.values[2])
    rankDictionary = nx.pagerank(G)
    ranks = list(rankDictionary.values())
    nodes = list(rankDictionary.keys())
    maximumRankedNode = nodes[ranks.index(max(ranks))]
    minimumRankedNode = nodes[ranks.index(min(ranks))]
    return maximumRankedNode

def get_graph(user):
    G = nx.Graph()
    for i,j in friends[friends.user == user].iterrows():
        G.add_edge(j.values[1],j.values[2])
    return G

cliqueSize = 5
tooLittleFriendsInCircleThreshold = 6
tooManyNodesThreshold = 315

egonetFolderName = 'data/egonets/'
submissionFolderName = 'submissions/'

def user_clique(user):
    
    G = get_graph(user)
    c = maxrank(user)
    
    print str(len(G.nodes())) + " nodes for user " + str(user)

    if user in [345,0,21869,18844]:
        return (user, [[c]])
    # do not calculate for large graphs (it takes too long)
    if len(G.nodes()) > tooManyNodesThreshold:
        return (user, [[c]])

    # find comunities using k_clique_communities()
    listOfCircles = []
    kCliqueComunities = list(nx.k_clique_communities(G,cliqueSize))
    for community in kCliqueComunities:
        # leave only relativly large communities
        if len(community) >= tooLittleFriendsInCircleThreshold:
            listOfCircles.append(list(community))

    # if no prediction was created, use max pagerank friend
    if len(listOfCircles) == 0:
        return (user, [[c]])
    else:
        return (user, listOfCircles)

def predict_circles(user):
    print user
    user, listOfCircles = user_clique(user)
    return (user, listOfCircles)

logging.info('Started')
logging.info('-train set')
predictionTrain = map(predict_circles, train)
logging.info('-test set')
prediction = map(predict_circles, test)
logging.info('Finished')

w = df(predictionTrain, columns=['UserId','Predicted'])

predictionTrain = []
for row in range(w.shape[0]):
    us = w.ix[row,'UserId']
    cs = w.ix[row,'Predicted']
    cs = [' '.join([str(y) for y in x]) for x in cs]
    predictionTrain.append(str(us) + ',' + ';'.join(cs))

w = '\n'.join(predictionTrain)

with open("submissions/benchmarkTrain.csv", "wb") as f:
    f.write('UserId,Predicted\n')
    f.write(w)

w = df(prediction, columns=['UserId','Predicted'])

prediction = []
for row in range(w.shape[0]):
    us = w.ix[row,'UserId']
    cs = w.ix[row,'Predicted']
    cs = [' '.join([str(y) for y in x]) for x in cs]
    prediction.append(str(us) + ',' + ';'.join(cs))

w = '\n'.join(prediction)

with open("submissions/benchmark.csv", "wb") as f:
    f.write('UserId,Predicted\n')
    f.write(w)
