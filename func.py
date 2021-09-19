import random
import numpy as np
import operator
from sklearn.cluster import KMeans
import dlib
from scipy.stats import spearmanr
from sklearn.metrics import recall_score
from math import ceil
import matplotlib.pyplot as plt
import json
import sys

global possible_values
global difficulties
global golden_ranking
global items_to_create
global G_size
global threshold
global uniformity
global iters

global num_of_workers
global spammers
global unskilled

global errors
global bias

#global time_randomness

#Read dataset from file.
if len(sys.argv)>1:
	with open(sys.argv[1]) as f:
		data = json.load(f)
	possible_values = data['possible_values']
	difficulties = data['difficulties']

for d in difficulties:
    difficulties[d] /= 4	#this number can be tweaked.
    
#to_json = {'possible_values':possible_values,
#			'difficulties':difficulties}
			
#with open('rainbow.json', 'w') as fp:
#    json.dump(to_json, fp)

golden_ranking = sorted(difficulties.items(), key=lambda kv: kv[1])
golden_ranking = np.array(golden_ranking)

#hyperparameters

items_to_create = 690   #fixed
G_size = 0.3  #fixed
threshold = 0.7 #fixed
uniformity = 0.2    #fixed
iters = 2 #fixed

num_of_workers = 50 #fixed
spammers = 0 #fixed
unskilled = 0 #fixed

errors = 0.05
bias = 1

#Basic functions

def create_items(n,uniformity): 
    items = []
    for i in range(n):
        item = {'id':i}
        for v in possible_values:
            color_index = np.random.normal(0.5,uniformity)
            thresholds = [i/11 for i in range(1,12)]
            min_dist = 10000
            best_index = -1
            for j in range(11):
                dist = abs(color_index - thresholds[j])
                if dist<min_dist:
                    min_dist = dist
                    best_index = j
            item[v] = list(possible_values[v])[best_index]
        items.append(item)
    return items

def create_workers(n, spammers, unskilled):	#spammers and unskilled are
											#ratios (e.g. 0.1)
    workers = []
    for i in range(n):
        worker_status = 0
        chance = random.random()
        if chance<spammers:
            worker_status = 1
        elif chance<unskilled:
            worker_status = 2
        workers.append(worker_status)
    return workers

def create_regions(items,field):#in our case, regions are colors.
								#All 'blue' questions are in one region.
    regions = {}
    for v in possible_values[field]:
        regions[v] = []
    for i in items:
        if field in i:
            regions[i[field]].append(i)
    return regions

#Simulate the crowdsourcing process.
def generate_answers(workers,num_of_items,num_of_answers,difficulty,time_randomness):
    #print(time_randomness)
    answers = np.ones((num_of_items,num_of_answers))
    times = np.zeros((num_of_items,num_of_answers))
    question_workers = []
    for i in range(num_of_items):
        chosen_workers = random.sample(range(len(workers)),k=num_of_answers)
        question_workers.append(chosen_workers)
        for j in range(num_of_answers):
            worker = workers[chosen_workers[j]]
            is_error = random.random()<errors
            times[i,j] = time_randomness * random.random() + (1-time_randomness)*difficulty
            if worker==1:    #spammer
                answers[i,j] = random.random()<0.5
                times[i,j] = time_randomness*random.random()/10
            elif worker==2:  #unskilled
                if random.random()<difficulty*2:
                    answers[i,j] = 0
            else:                   #good worker
                if random.random()<difficulty:
                    answers[i,j] = 0
    return answers,times,question_workers

#Analyze the crowdsourcing results per region
def generate_region_metrics(workers,num_of_items,num_of_answers,difficulty,
                            time_randomness=0.5):
    
    answers,times,question_workers = generate_answers(workers,num_of_items,num_of_answers,difficulty,time_randomness)
    accuracy = np.mean(answers)

    total_answers = num_of_items*num_of_answers
    ones = accuracy*total_answers
    disagreement = 1 - abs(ones - (total_answers-ones))/(total_answers)
    time = np.mean(times)
    return accuracy,time,disagreement,answers,times,question_workers

#use a horizontal baseline plan to determine each region's difficulty ranking.
def generate_regions_baseline(workers,regions,x,y):
    baseline_difficulties = {}
    costs = 0
    for r in regions:
        region = regions[r]

        costs += x*(y+10)
        accuracy = generate_region_metrics(workers,x,y,difficulties[r])[0]
        baseline_difficulties[r] = 1-accuracy
    
    sorted_colors = sorted(baseline_difficulties.items(), key=lambda kv: kv[1])
    return np.array(sorted_colors), costs
