import random
import numpy as np
import matplotlib.pyplot as plt

from ranking import *
from finding_plan import *

#json reading and hyperparameters are in func.py

#rank the regions by difficulty and
#find the best value for time_randomness.
#The necessary variables are set as global.

time_randomness, metrics, b_metrics = do_the_ranking(plot=True)
print("The most realistic value for time randomness is", time_randomness)
print("The precision, recall, and F1-score of the rankings compared to \
the true rankings, are:")
print("Baseline:", b_metrics)
print("Optimal :", metrics)

time_randomness = 0.1
b_costs, b_accs, costs, accs = find_best_plan_pipeline(time_randomness)

for i in range(len(accs)):
	accs[i] = round(accs[i],3)
for i in range(len(b_accs)):
	b_accs[i] = round(b_accs[i],3)

print("\nAgainst baseline plans that use 3,5,7, and 9 answer per question,\
the respective plans found by VChecker are:")
print("Baseline costs are:",b_costs)
print("Baseline accuracies are:",b_accs)
print("Costs are:",costs)
print("Accuracies are:",accs)

reds = []
for i in range(4):
	reds.append(round(100*(b_costs[i]-costs[i])/b_costs[i],2))
print("\nVChecker achieves a cost reduction of (in %):",reds)




#for i in range(4):
#	print("For a baseline plan of", 3+2*i, "answer per question,\
#	 VChecker achieves a cost reduction of",100*(b_costs[i]-costs[i])/b_costs[i],\
#	 "\%")
