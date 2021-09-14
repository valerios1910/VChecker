import random
import numpy as np
import operator
from sklearn.cluster import KMeans
import dlib
from scipy.stats import spearmanr
from sklearn.metrics import recall_score
from math import ceil
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

from core_funcs import *
from func import *
from ranking import *

def f(i,t,a):  #the names are as defined in the paper.
    value = 0
    for j in range(ceil(t/2),t+1):
        value += len(list(combinations(range(t),j))) * pow(a[i],j) * pow(1-a[i],t-j)
    return value
    
def optimize_crowdsourcing(ranking,region_metrics,t_b,regions,x,y,workers,plan=0):
    #create the initial variables
    n = [len(regions[color]) for color in ranking[:,0]]
    d = [difficulties[color] for color in ranking[:,0]]
    a = [region_metrics[color][0] for color in ranking[:,0]]

    m = [n[i]*f(i,t_b,a) for i in range(len(n))]
    a_b = float(sum(m)) / sum(n)

    #adjust accuracies according to ranking
    m = gp.Model("qp")
    m.setParam('OutputFlag', 0)
    z = m.addVars(11)
    m.setObjective(sum((z[i]-a[i])*(z[i]-a[i]) for i in z.keys()), GRB.MINIMIZE)
    for i in range(len(a)):
        m.addConstr(0<=z[i], "cl"+str(i))
        m.addConstr(z[i]<=1, "cu"+str(i))
        for j in range(i,len(a)):
            m.addConstr(z[i]>=z[j], "cc"+str(j))
    m.optimize()

    for i in range(len(a)):
        a[i] = z[i].x

    #solve the optimization problem
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    h = m.addMVar((len(list(regions)),9), vtype=GRB.BINARY)

    if plan==0:
        m.setObjective(sum(x*y+sum((2*j+1)*h[i,j] for j in range(9))*(n[i]-x) for i in range(len(list(h)))), GRB.MINIMIZE)

        m.addConstr(sum(x+(n[i]-x)*sum(h[i,j]*f(i,2*j+1,a) for j in range(9)) for i in range(len(list(h))))/sum(n) >= a_b, "c1")
        m.addConstr(sum(x*y+sum((2*j+1)*h[i,j] for j in range(9))*(n[i]-x) for i in range(len(list(h)))) <= t_b*sum(n), "c2")
        for i in range(len(list(h))):
            m.addConstr(sum(h[i])==1,"c3"+str(i))
    elif plan==1:
        m.setObjective(sum(x+(n[i]-x)*sum(h[i,j]*f(i,2*j+1,a) for j in range(9)) for i in range(len(list(h))))/sum(n), GRB.MAXIMIZE)

        m.addConstr(sum(x+(n[i]-x)*sum(h[i,j]*f(i,2*j+1,a) for j in range(9)) for i in range(len(list(h))))/sum(n) >= a_b, "c1")
        m.addConstr(sum(x*y+sum((2*j+1)*h[i,j] for j in range(9))*(n[i]-x) for i in range(len(list(h)))) <= t_b*sum(n), "c2")
        for i in range(len(list(h))):
            m.addConstr(sum(h[i])==1,"c3"+str(i))
    else:
        print("Invalid plan.")
        return None

    m.optimize()

    #add the t for each region
    t = []
    if m.status == GRB.OPTIMAL:
        i = 0
        for v in m.getVars():
            if v.x==1:
                t.append(2*i+1)
            i+=1
            if i==9:
                i=0
    else:
        t = [t_b for i in range(len(list(regions)))]

    #calculate the actual accuracy
    reg_accs = []
    for i in range(len(t)):
        reg_accs.append(generate_region_metrics(workers,n[i],t[i],d[i])[0])

    b_reg_accs = []
    for i in range(len(t)):
        b_reg_accs.append(generate_region_metrics(workers,n[i],t_b,d[i])[0])

    #return results
    b_act_acc = np.mean(b_reg_accs)    
    act_acc=np.mean(reg_accs)
    cost = sum(x*y+t[i]*(n[i]-x) for i in range(len(t)))
    acc = sum(x+(n[i]-x)*f(i,t[i],a) for i in range(len(t)))/sum(n)
    b_cost = t_b*sum(n)
    b_acc = a_b

    return cost,acc,act_acc,b_cost,b_acc,b_act_acc

def find_best_plan_pipeline(time_randomness):
	b_costss = []
	b_accss = []
	costss = []
	accss = []
	
	for t_b in [3,5,7,9]:
		b_costs = 0
		b_accs = 0
		costs = 0
		accs = 0

		for _ in range(iters):
			workers = create_workers(num_of_workers,spammers,unskilled)
			items = create_items(items_to_create,uniformity)
			regions = create_regions(items,'color')

			x,y = find_x_y(workers,regions,time_randomness,threshold)
			ranking,b_ranking,region_metrics = rank_regions(workers,time_randomness,G_size,regions,x,y)

			cost,acc,act_acc,b_cost,b_acc,b_act_acc = optimize_crowdsourcing(ranking,
				region_metrics,t_b,regions,x,y,workers,plan=0)
			b_costs += b_cost
			b_accs += b_act_acc
			accs+=act_acc
			costs+=cost

		b_costss.append(b_costs/iters)
		b_accss.append(b_accs/iters)
		costss.append(costs/iters)
		accss.append(accs/iters)
		
	return b_costss, b_accss, costss, accss
