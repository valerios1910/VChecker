import random
import numpy as np
import operator
from sklearn.cluster import KMeans
import dlib
from scipy.stats import spearmanr
from sklearn.metrics import recall_score
from math import ceil
import matplotlib.pyplot as plt

from core_funcs import *
from func import *

#create the sets needed for ranking calibration
def create_sets(ranking):
    values = ranking[:,0]
    sets = []
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            sets.append((values[i],values[j]))
    return sets

#compare ranking sets
def compare_ranking_sets(set_true,set_pred):
    precision = 0
    for s in set_pred:
        if s in set_true:
            precision+=1
    precision/=len(set_pred)

    recall = 0
    for s in set_true:
        if s in set_pred:
            recall+=1
    recall/=len(set_pred)

    f1 = 2*precision*recall/(precision+recall)

    return precision,recall,f1
    
#do the final ranking of regions. Similar to 'find_ranking', but
#they are used for slightly different purposes.
def rank_regions(workers,time_randomness,G_size,regions,x,y):
    results = {}

    baseline_ranking, baseline_cost = generate_regions_baseline(workers,regions,x,y)

    region_metrics = {}
    for r in regions:
        region_metrics[r] = generate_region_metrics(workers,x,y,difficulties[r],
                                            time_randomness=time_randomness)

    region_metrics_2 = region_metrics.copy()
    for r in region_metrics_2:
        region_metrics_2[r] = region_metrics_2[r][:3]
    ranking = train_SVM_rank(region_metrics_2,G_size=G_size,tests=0)

    return ranking,baseline_ranking,region_metrics

#perform the pipeline of ranking, then discover the best value for the
#'time randomness' hyperparameter.
def do_the_ranking(plot=True):
	metricsss = []
	b_metricsss = []
	xss = []
	yss = []
	time_randomness = 0.1
	while time_randomness<=1:
		metricss = [0,0,0]
		b_metricss = [0,0,0]
		xs = ys = 0

		for _ in range(iters):
			workers = create_workers(num_of_workers,spammers,unskilled)
			items = create_items(items_to_create,uniformity)
			regions = create_regions(items,'color')

			x,y = find_x_y(workers,regions,time_randomness,threshold)
			xs+=x
			ys+=y
			ranking,b_ranking,region_metrics = rank_regions(workers,time_randomness,G_size,regions,x,y)

			golden_sets = create_sets(golden_ranking)
			sets = create_sets(ranking)
			b_sets = create_sets(b_ranking)

			b_metrics = compare_ranking_sets(golden_sets,b_sets)
			metrics = compare_ranking_sets(golden_sets,sets)

			for i in range(3):
				metricss[i]+=metrics[i]
				b_metricss[i]+=b_metrics[i]

		metricsss.append([metricss[i]/float(iters) for i in range(3)])
		b_metricsss.append([b_metricss[i]/float(iters) for i in range(3)])
		xss.append(xs/iters)
		yss.append(ys/iters)

		time_randomness+=0.1
		
	if plot:
		times = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
		plt.plot(times,np.array(metricsss)[:,2],label="VChecker")
		plt.plot(times,np.array(b_metricsss)[:,2],label="WAK")

		plt.title('time randomness vs Average ranking accuracy')
		plt.xlabel('time_randomness')
		plt.ylabel('F1-score')
		plt.legend(loc='best')
		plt.show()
	
	best_time_randomness = 0.1*(np.argmax(metricsss[:][2])+1)
	metrics = metricsss[np.argmax(metricsss[:][2])]
	b_metrics = b_metricsss[np.argmax(metricsss[:][2])]
	
	return best_time_randomness, metrics, b_metrics
