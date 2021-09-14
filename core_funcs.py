import random
import numpy as np
import operator
from sklearn.cluster import KMeans
import dlib
from scipy.stats import spearmanr
from sklearn.metrics import recall_score
from math import ceil
import matplotlib.pyplot as plt

from func import *

#Generate the region pairs that will train the SVM
def generate_training_pairs(region_metrics,G_size=0.25):
    region_metrics_list = list(region_metrics)
    G_size = int(len(region_metrics)*G_size)
    G_indexes = random.sample(region_metrics_list,G_size)
    #print(G_indexes)
    G_difficulties = [difficulties[g] for g in G_indexes]
    #print(G_difficulties)

    G = [region_metrics[g] for g in G_indexes]

    kmeans = KMeans(n_clusters=2, random_state=42).fit(G)

    ds = [0,0]
    ones=0
    for i in range(len(G_difficulties)):
        ds[kmeans.labels_[i]]+=difficulties[G_indexes[i]]
        ones+=kmeans.labels_[i]
    ds[1] /= ones
    ds[0] /= (G_size-ones)

    difficult_group = ds.index(max(ds))

    training_pairs = [(G[i],1) if kmeans.labels_[i]==difficult_group else (G[i],2) for i in range(len(G))]
    for_debugging = [(G_indexes[i],1) if kmeans.labels_[i]==difficult_group else (G_indexes[i],2) for i in range(len(G_indexes))]

    error = 0
    for g1 in for_debugging:
        for g2 in for_debugging:
            if g1[1]>g2[1] and difficulties[g1[0]]>difficulties[g2[0]]:
                error=1
                break

    return training_pairs, error

#train an SVM regressor to rank the regions by difficulty
def train_SVM_rank(region_metrics,G_size=0.25,tests=0):
    train_pairs=None
    if tests==0:
        train_pairs,_err = generate_training_pairs(region_metrics,G_size)
    else:
        errors = 0
        for _ in range(tests):
            train_pairs,err = generate_training_pairs(region_metrics,G_size)
            errors+=err
        print("The training pairs are", errors*10/tests, "% inaccurate.")
    data = dlib.ranking_pair()
    for tp in train_pairs:
        if tp[1]==1:
            data.relevant.append(dlib.vector(tp[0]))
        else:
            data.nonrelevant.append(dlib.vector(tp[0]))

    trainer = dlib.svm_rank_trainer()
    rank = trainer.train(data)
    scores = {}
    for r in region_metrics:
        scores[r] = rank(dlib.vector(region_metrics[r]))
    
    ranking = sorted(scores.items(), key=lambda kv: kv[1])
    return np.array(ranking)

#streamline the SVM training
def find_ranking(workers,regions,time_randomness,x,y):
    region_metrics = {}
    for r in regions:
        region_metrics[r] = generate_region_metrics(workers,x,y,difficulties[r],
                                            time_randomness)[:3]

    ranking = train_SVM_rank(region_metrics,tests=0)
    return np.array(ranking)
    
#find the optimal x (number of questions) and y (number of answers per question)
#for the initial crowdsourcing. This is a crowdsourcing on a small
#subset of our dataset to calibrate the SVM ranker.
def find_x_y(workers,regions,time_randomness,threshold=0.7):
    rankings = [find_ranking(workers,regions,time_randomness,2,2)[:,0]]
    rankings.append(find_ranking(workers,regions,time_randomness,2,3)[:,0])
    x = y = 1

    while x<=5 and y<=5:
        rankings.append(find_ranking(workers,regions,time_randomness,x,y)[:,0])
        if abs(spearmanr(rankings[-1],rankings[-2]).correlation) > threshold:
            if abs(spearmanr(rankings[-2],rankings[-3]).correlation) > threshold:
                break
        if x==y:
            y+=1
        else:
            x+=1

    return x,y
