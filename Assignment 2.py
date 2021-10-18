# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:52:02 2021

@author: opena8
"""

"""
Code used to define the optimization functions was adapted from the one found in the 
ml rose online documentation and examples.
https://mlrose.readthedocs.io/en/stable/source/intro.html
"""



import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import time as time
import random
import pandas as pd
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

seed=0
random.seed(seed)

def plot_fitness(best_state, alg, best_fitness, fitness_curve):
    print("Using {}".format(alg))
    print("The best state found by is: ", best_state)
    print("The fitness at the best state is: ", best_fitness)
    plt.title("Fitness Curve - {}".format(alg))
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.plot(fitness_curve[:,1], fitness_curve[:,0])
    plt.show()
    
    
############################################################# SIX PEAKS PROBLEM #############################################################
fitness_function=mr.SixPeaks(t_pct=0.2)
size=100
init_state=[]

for i in range(0,size,1):
    init_state.append(random.randint(0,1))
print(init_state)

problem=mr.DiscreteOpt(length=size, fitness_fn=fitness_function, maximize=True, max_val=2)

#Create lists to hold the values inside the loops
peaks_best_states_rhc=[]
peaks_best_fitnesses_rhc=[]
peaks_best_fitness_curves_rhc=[]

peaks_best_states_sa=[]
peaks_best_fitnesses_sa=[]
peaks_best_fitness_curves_sa=[]

peaks_best_states_ga=[]
peaks_best_fitnesses_ga=[]
peaks_best_fitness_curves_ga=[]

peaks_best_states_mimic=[]
peaks_best_fitnesses_mimic=[]
peaks_best_fitness_curves_mimic=[]

#Try multiple values for restarts of the random hill climbing to find best hyperparameter
start=time.time()
for i in range(1, 101, 1):
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc=mr.random_hill_climb(problem, max_attempts=20, init_state = init_state, random_state=0, curve=True, restarts=i) 
    peaks_best_states_rhc.append(best_state_rhc)
    peaks_best_fitnesses_rhc.append(best_fitness_rhc)
    peaks_best_fitness_curves_rhc.append(fitness_curve_rhc)

    
peaks_best_fitness_final_rhc=max(peaks_best_fitnesses_rhc)
peaks_index_best_fitness_rhc=peaks_best_fitnesses_rhc.index(peaks_best_fitness_final_rhc)
peaks_best_state_rhc_final=peaks_best_states_rhc[peaks_index_best_fitness_rhc]
peaks_best_fitness_curve_rhc_final=peaks_best_fitness_curves_rhc[peaks_index_best_fitness_rhc]
end=time.time()
final_time=end-start
print("Time for RHC in Peaks ", final_time)

print("Maximum value for Random Hill Climbing", peaks_best_fitness_final_rhc, " occurs at index " , peaks_index_best_fitness_rhc)
print("Best State")
print(peaks_best_state_rhc_final)
plt.title("Six Peaks Fitness Curve - {}".format("Random Hill Climbing"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(peaks_best_fitness_curve_rhc_final[:,1], peaks_best_fitness_curve_rhc_final[:,0])
plt.show()

#Try multiple values for the temperature and find the best one for the problem 
start=time.time()
for i in range(10, 501 , 1):
    schedule=mr.ExpDecay(init_temp=i)
    best_state_sa, best_fitness_sa, fitness_curve_sa= mr.simulated_annealing(problem, max_attempts=20, init_state = init_state, schedule = schedule, random_state = 0, curve=True)
    peaks_best_states_sa.append(best_state_sa)
    peaks_best_fitnesses_sa.append(best_fitness_sa)
    peaks_best_fitness_curves_sa.append(fitness_curve_sa)
    

peaks_best_fitness_final_sa=max(peaks_best_fitnesses_sa)
peaks_index_best_fitness_sa=peaks_best_fitnesses_sa.index(peaks_best_fitness_final_sa)
peaks_best_state_sa_final=peaks_best_states_sa[peaks_index_best_fitness_sa]
peaks_best_fitness_curve_sa_final=peaks_best_fitness_curves_sa[peaks_index_best_fitness_sa]
end=time.time()
final_time=end-start
print("Time for Simulated Annealing in Peaks ", final_time)

print("Maximum value for Simulated Annealing", peaks_best_fitness_final_sa, " occurs at index " , peaks_index_best_fitness_sa)
print("Best State")
print(peaks_best_state_sa_final)
plt.title("Six Peaks Fitness Curve - {}".format("Simulated Annealing"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(peaks_best_fitness_curve_sa_final[:,1], peaks_best_fitness_curve_sa_final[:,0])
plt.show()

#Try multiple values for the population and find the best one for the problem 
start=time.time()
for i in range(50, 1050, 50):
    best_state_ga, best_fitness_ga, fitness_curve_ga=mr.genetic_alg(problem, random_state=0, curve=True, pop_size=i)
    peaks_best_states_ga.append(best_state_ga)
    peaks_best_fitnesses_ga.append(best_fitness_ga)
    peaks_best_fitness_curves_ga.append(fitness_curve_ga)
    

peaks_best_fitness_final_ga=max(peaks_best_fitnesses_ga)
peaks_index_best_fitness_ga=peaks_best_fitnesses_ga.index(peaks_best_fitness_final_ga)
peaks_best_state_ga_final=peaks_best_states_ga[peaks_index_best_fitness_ga]
peaks_best_fitness_curve_ga_final=peaks_best_fitness_curves_ga[peaks_index_best_fitness_ga]

end=time.time()
final_time=end-start
print("Time for Genetic Algorithm in Peaks ", final_time)

print("Maximum value for Genetic Algorithm", peaks_best_fitness_final_ga, " occurs at index " , peaks_index_best_fitness_ga)
print("Best State")
print(peaks_best_state_ga_final)
plt.title("Six Peaks Fitness Curve - {}".format("Genetic Algorithm"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(peaks_best_fitness_curve_ga_final[:,1], peaks_best_fitness_curve_ga_final[:,0])
plt.show()

#Try multiple values for the population and find the best one for the problem 
start=time.time()
for i in np.arange(.1, .3 , .05):
    print("Current iteration: {}".format(i))
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic=mr.mimic(problem, max_attempts=20, random_state=0, curve=True, keep_pct=i)
    peaks_best_states_mimic.append(best_state_mimic)
    peaks_best_fitnesses_mimic.append(best_fitness_mimic)
    peaks_best_fitness_curves_mimic.append(fitness_curve_mimic)
    

peaks_best_fitness_final_mimic=max(peaks_best_fitnesses_mimic)
peaks_index_best_fitness_mimic=peaks_best_fitnesses_mimic.index(peaks_best_fitness_final_mimic)
peaks_best_state_mimic_final=peaks_best_states_mimic[peaks_index_best_fitness_mimic]
peaks_best_fitness_curve_mimic_final=peaks_best_fitness_curves_mimic[peaks_index_best_fitness_mimic]

end=time.time()
final_time=end-start
print("Time for MIMIC in Peaks ", final_time)

print("Maximum value for MIMIC", peaks_best_fitness_final_mimic, " occurs at index " , peaks_index_best_fitness_mimic)
print("Best State")
print(peaks_best_state_mimic_final)
plt.title("Six Peaks Fitness Curve - {}".format("MIMIC"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(peaks_best_fitness_curve_mimic_final[:,1], peaks_best_fitness_curve_mimic_final[:,0])
plt.show()

plt.title("Fitness Curve - {}".format("All"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(peaks_best_fitness_curve_rhc_final[:,1], peaks_best_fitness_curve_rhc_final[:,0])
plt.plot(peaks_best_fitness_curve_sa_final[:,1], peaks_best_fitness_curve_sa_final[:,0])
plt.plot(peaks_best_fitness_curve_ga_final[:,1], peaks_best_fitness_curve_ga_final[:,0])
plt.plot(peaks_best_fitness_curve_mimic_final[:,1], peaks_best_fitness_curve_mimic_final[:,0])
plt.show()

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(peaks_best_fitness_curve_rhc_final[:,1], peaks_best_fitness_curve_rhc_final[:,0], color="black", linewidth=1, label='RHC')
ax1.set_ylabel("Fitness")
ax1.set_xlabel("Iteration")
ax1.set_ylim([0, 200])
ax1.legend(loc="upper left")

ax2.plot(peaks_best_fitness_curve_sa_final[:,1], peaks_best_fitness_curve_sa_final[:,0],color="blue", linewidth=1, label='SA')
ax2.set_ylabel("Fitness")
ax2.set_xlabel("Iteration")
ax2.set_ylim([0, 200])
ax2.legend(loc="upper left")

ax3.plot(peaks_best_fitness_curve_ga_final[:,1], peaks_best_fitness_curve_ga_final[:,0], color="red", linewidth=1, label='GA')
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Iteration")
ax3.set_ylim([0, 200])
ax3.legend(loc="upper left")

ax4.plot(peaks_best_fitness_curve_mimic_final[:,1], peaks_best_fitness_curve_mimic_final[:,0], color="orange", linewidth=1, label='MIMIC')
ax4.set_ylabel("Fitness")
ax4.set_xlabel("Iteration")
ax4.set_ylim([0, 200])
ax4.legend(loc="upper left")

figure.suptitle("Comparison Between RHC, SA, GA, MIMIC - Six Peaks Problem")

plt.tight_layout()

plt.show()
############################################################# N-QUEENS #############################################################  
print(" ")
print("THis is the start of the N-QUEENS PROBLEM")

#Code copied from mlrose documentation. Used to implement a maximization fitness function version of N-Queens
#https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
def queens_max(state):
# Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                   # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

# Initialize custom fitness function object
fitness_function = mr.CustomFitness(queens_max)

schedule=mr.ExpDecay()
size=20
init_state=np.arange(0,size,1)
problem=mr.DiscreteOpt(length=size, fitness_fn=fitness_function, maximize=True, max_val=size)

#Create lists to hold the values inside the loops
queens_best_states_rhc=[]
queens_best_fitnesses_rhc=[]
queens_best_fitness_curves_rhc=[]

queens_best_states_sa=[]
queens_best_fitnesses_sa=[]
queens_best_fitness_curves_sa=[]

queens_best_states_ga=[]
queens_best_fitnesses_ga=[]
queens_best_fitness_curves_ga=[]

queens_best_states_mimic=[]
queens_best_fitnesses_mimic=[]
queens_best_fitness_curves_mimic=[]

#Try multiple values for restarts of the random hill climbing to find best hyperparameter
start=time.time()
for i in range(1, 101, 1):
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc=mr.random_hill_climb(problem, max_attempts=20, init_state = init_state, random_state=0, curve=True, restarts=i) 
    queens_best_states_rhc.append(best_state_rhc)
    queens_best_fitnesses_rhc.append(best_fitness_rhc)
    queens_best_fitness_curves_rhc.append(fitness_curve_rhc)

    
queens_best_fitness_final_rhc=max(queens_best_fitnesses_rhc)
queens_index_best_fitness_rhc=queens_best_fitnesses_rhc.index(queens_best_fitness_final_rhc)
queens_best_state_rhc_final=queens_best_states_rhc[queens_index_best_fitness_rhc]
queens_best_fitness_curve_rhc_final=queens_best_fitness_curves_rhc[queens_index_best_fitness_rhc]
end=time.time()
final_time=end-start
print("Time for RHC in Queens ", final_time)

print("Maximum value for Random Hill Climbing", queens_best_fitness_final_rhc, " occurs at index " , queens_index_best_fitness_rhc)
print("Best State")
print(queens_best_state_rhc_final)
plt.title("N-Queens Fitness Curve - {}".format("Random Hill Climbing"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(queens_best_fitness_curve_rhc_final[:,1], queens_best_fitness_curve_rhc_final[:,0])
plt.show()

#Try multiple values for the temperature and find the best one for the problem 
start=time.time()
for i in range(10, 501 , 1):
    schedule=mr.ExpDecay(init_temp=i)
    best_state_sa, best_fitness_sa, fitness_curve_sa= mr.simulated_annealing(problem, max_attempts=20, init_state = init_state, schedule = schedule, random_state = 0, curve=True)
    queens_best_states_sa.append(best_state_sa)
    queens_best_fitnesses_sa.append(best_fitness_sa)
    queens_best_fitness_curves_sa.append(fitness_curve_sa)
    

queens_best_fitness_final_sa=max(queens_best_fitnesses_sa)
queens_index_best_fitness_sa=queens_best_fitnesses_sa.index(queens_best_fitness_final_sa)
queens_best_state_sa_final=queens_best_states_sa[queens_index_best_fitness_sa]
queens_best_fitness_curve_sa_final=queens_best_fitness_curves_sa[queens_index_best_fitness_sa]
end=time.time()
final_time=end-start
print("Time for Simulated Annealing in Queens ", final_time)

print("Maximum value for Simulated Annealing", queens_best_fitness_final_sa, " occurs at index " , queens_index_best_fitness_sa)
print("Best State")
print(queens_best_state_sa_final)
plt.title("N-Queens Fitness Curve - {}".format("Simulated Annealing"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(queens_best_fitness_curve_sa_final[:,1], queens_best_fitness_curve_sa_final[:,0])
plt.show()

#Try multiple values for the population and find the best one for the problem 
start=time.time()
for i in range(50, 1050, 50):
    best_state_ga, best_fitness_ga, fitness_curve_ga=mr.genetic_alg(problem, random_state=0, curve=True, pop_size=i)
    queens_best_states_ga.append(best_state_ga)
    queens_best_fitnesses_ga.append(best_fitness_ga)
    queens_best_fitness_curves_ga.append(fitness_curve_ga)
    

queens_best_fitness_final_ga=max(queens_best_fitnesses_ga)
queens_index_best_fitness_ga=queens_best_fitnesses_ga.index(queens_best_fitness_final_ga)
queens_best_state_ga_final=queens_best_states_ga[queens_index_best_fitness_ga]
queens_best_fitness_curve_ga_final=queens_best_fitness_curves_ga[queens_index_best_fitness_ga]

end=time.time()
final_time=end-start
print("Time for Genetic Algorithm in Queens ", final_time)

print("Maximum value for Genetic Algorithm", queens_best_fitness_final_ga, " occurs at index " , queens_index_best_fitness_ga)
print("Best State")
print(queens_best_state_ga_final)
plt.title("N-Queens Fitness Curve - {}".format("Genetic Algorithm"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(queens_best_fitness_curve_ga_final[:,1], queens_best_fitness_curve_ga_final[:,0])
plt.show()

#Try multiple values for the population and find the best one for the problem 
start=time.time()
for i in np.arange(.1, .3 , .05):
    print("Current iteration: {}".format(i))
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic=mr.mimic(problem, max_attempts=20, random_state=0, curve=True, keep_pct=i)
    queens_best_states_mimic.append(best_state_mimic)
    queens_best_fitnesses_mimic.append(best_fitness_mimic)
    queens_best_fitness_curves_mimic.append(fitness_curve_mimic)
    

queens_best_fitness_final_mimic=max(queens_best_fitnesses_mimic)
queens_index_best_fitness_mimic=queens_best_fitnesses_mimic.index(queens_best_fitness_final_mimic)
queens_best_state_mimic_final=queens_best_states_mimic[queens_index_best_fitness_mimic]
queens_best_fitness_curve_mimic_final=queens_best_fitness_curves_mimic[queens_index_best_fitness_mimic]

end=time.time()
final_time=end-start
print("Time for MIMIC in Queens ", final_time)

print("Maximum value for MIMIC", queens_best_fitness_final_mimic, " occurs at index " , queens_index_best_fitness_mimic)
print("Best State")
print(queens_best_state_mimic_final)
plt.title("N-Queens Fitness Curve - {}".format("MIMIC"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(queens_best_fitness_curve_mimic_final[:,1], queens_best_fitness_curve_mimic_final[:,0])
plt.show()

plt.title("Fitness Curve - {}".format("All"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(queens_best_fitness_curve_rhc_final[:,1], queens_best_fitness_curve_rhc_final[:,0])
plt.plot(queens_best_fitness_curve_sa_final[:,1], queens_best_fitness_curve_sa_final[:,0])
plt.plot(queens_best_fitness_curve_ga_final[:,1], queens_best_fitness_curve_ga_final[:,0])
plt.plot(queens_best_fitness_curve_mimic_final[:,1], queens_best_fitness_curve_mimic_final[:,0])
plt.show()


figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(queens_best_fitness_curve_rhc_final[:,1], queens_best_fitness_curve_rhc_final[:,0], color="black", linewidth=1, label='RHC')
ax1.set_ylabel("Fitness")
ax1.set_xlabel("Iteration")
ax1.set_ylim([0, 200])
ax1.legend(loc="upper left")

ax2.plot(queens_best_fitness_curve_sa_final[:,1], queens_best_fitness_curve_sa_final[:,0],color="blue", linewidth=1, label='SA')
ax2.set_ylabel("Fitness")
ax2.set_xlabel("Iteration")
ax2.set_ylim([0, 200])
ax2.legend(loc="upper left")

ax3.plot(queens_best_fitness_curve_ga_final[:,1], queens_best_fitness_curve_ga_final[:,0], color="red", linewidth=1, label='GA')
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Iteration")
ax3.set_ylim([0, 200])
ax3.legend(loc="upper left")

ax4.plot(queens_best_fitness_curve_mimic_final[:,1], queens_best_fitness_curve_mimic_final[:,0], color="orange", linewidth=1, label='MIMIC')
ax4.set_ylabel("Fitness")
ax4.set_xlabel("Iteration")
ax4.set_ylim([0, 200])
ax4.legend(loc="upper left")

figure.suptitle("Comparison Between RHC, SA, GA, MIMIC - N Queens Problem")

plt.tight_layout()
plt.show()
############################################################# FLIP FLOP #############################################################  
print(" ")
print("THis is the start of the FLIP FLOP PROBLEM")
fitness_function=mr.FlipFlop()
schedule=mr.ExpDecay()
size=100

init_state=[]
for i in range(0,size,1):
    init_state.append(random.randint(0,1))
problem=mr.DiscreteOpt(length=size, fitness_fn=fitness_function, maximize=True, max_val=2)

#Create lists to hold the values inside the loops
flip_flop_best_states_rhc=[]
flip_flop_best_fitnesses_rhc=[]
flip_flop_best_fitness_curves_rhc=[]

flip_flop_best_states_sa=[]
flip_flop_best_fitnesses_sa=[]
flip_flop_best_fitness_curves_sa=[]

flip_flop_best_states_ga=[]
flip_flop_best_fitnesses_ga=[]
flip_flop_best_fitness_curves_ga=[]

flip_flop_best_states_mimic=[]
flip_flop_best_fitnesses_mimic=[]
flip_flop_best_fitness_curves_mimic=[]

#Try multiple values for restarts of the random hill climbing to find best hyperparameter
start=time.time()
for i in range(1, 101, 1):
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc=mr.random_hill_climb(problem, max_attempts=20, init_state = init_state, random_state=0, curve=True, restarts=i) 
    flip_flop_best_states_rhc.append(best_state_rhc)
    flip_flop_best_fitnesses_rhc.append(best_fitness_rhc)
    flip_flop_best_fitness_curves_rhc.append(fitness_curve_rhc)

    
flip_flop_best_fitness_final_rhc=max(flip_flop_best_fitnesses_rhc)
flip_flop_index_best_fitness_rhc=flip_flop_best_fitnesses_rhc.index(flip_flop_best_fitness_final_rhc)
flip_flop_best_state_rhc_final=flip_flop_best_states_rhc[flip_flop_index_best_fitness_rhc]
flip_flop_best_fitness_curve_rhc_final=flip_flop_best_fitness_curves_rhc[flip_flop_index_best_fitness_rhc]
end=time.time()
final_time=end-start
print("Time for RHC in Flip Flop ", final_time)

print("Maximum value for Random Hill Climbing", flip_flop_best_fitness_final_rhc, " occurs at index " , flip_flop_index_best_fitness_rhc)
print("Best State")
print(flip_flop_best_state_rhc_final)
plt.title("N-Flip Flop Fitness Curve - {}".format("Random Hill Climbing"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(flip_flop_best_fitness_curve_rhc_final[:,1], flip_flop_best_fitness_curve_rhc_final[:,0])
plt.show()

#Try multiple values for the temperature and find the best one for the problem 
start=time.time()
for i in range(10, 501 , 1):
    schedule=mr.ExpDecay(init_temp=i)
    best_state_sa, best_fitness_sa, fitness_curve_sa= mr.simulated_annealing(problem, max_attempts=20, init_state = init_state, schedule = schedule, random_state = 0, curve=True)
    flip_flop_best_states_sa.append(best_state_sa)
    flip_flop_best_fitnesses_sa.append(best_fitness_sa)
    flip_flop_best_fitness_curves_sa.append(fitness_curve_sa)
    

flip_flop_best_fitness_final_sa=max(flip_flop_best_fitnesses_sa)
flip_flop_index_best_fitness_sa=flip_flop_best_fitnesses_sa.index(flip_flop_best_fitness_final_sa)
flip_flop_best_state_sa_final=flip_flop_best_states_sa[flip_flop_index_best_fitness_sa]
flip_flop_best_fitness_curve_sa_final=flip_flop_best_fitness_curves_sa[flip_flop_index_best_fitness_sa]
end=time.time()
final_time=end-start
print("Time for Simulated Annealing in Flip Flop ", final_time)

print("Maximum value for Simulated Annealing", flip_flop_best_fitness_final_sa, " occurs at index " , flip_flop_index_best_fitness_sa)
print("Best State")
print(flip_flop_best_state_sa_final)
plt.title("N-Flip Flop Fitness Curve - {}".format("Simulated Annealing"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(flip_flop_best_fitness_curve_sa_final[:,1], flip_flop_best_fitness_curve_sa_final[:,0])
plt.show()

#Try multiple values for the population and find the best one for the problem 
start=time.time()
for i in range(50, 1050, 50):
    best_state_ga, best_fitness_ga, fitness_curve_ga=mr.genetic_alg(problem, random_state=0, curve=True, pop_size=i)
    flip_flop_best_states_ga.append(best_state_ga)
    flip_flop_best_fitnesses_ga.append(best_fitness_ga)
    flip_flop_best_fitness_curves_ga.append(fitness_curve_ga)
    

flip_flop_best_fitness_final_ga=max(flip_flop_best_fitnesses_ga)
flip_flop_index_best_fitness_ga=flip_flop_best_fitnesses_ga.index(flip_flop_best_fitness_final_ga)
flip_flop_best_state_ga_final=flip_flop_best_states_ga[flip_flop_index_best_fitness_ga]
flip_flop_best_fitness_curve_ga_final=flip_flop_best_fitness_curves_ga[flip_flop_index_best_fitness_ga]

end=time.time()
final_time=end-start
print("Time for Genetic Algorithm in Flip Flop ", final_time)

print("Maximum value for Genetic Algorithm", flip_flop_best_fitness_final_ga, " occurs at index " , flip_flop_index_best_fitness_ga)
print("Best State")
print(flip_flop_best_state_ga_final)
plt.title("N-Flip Flop Fitness Curve - {}".format("Genetic Algorithm"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(flip_flop_best_fitness_curve_ga_final[:,1], flip_flop_best_fitness_curve_ga_final[:,0])
plt.show()

#Try multiple values for the population and find the best one for the problem 
start=time.time()
for i in np.arange(.1, .3 , .05):
    print("Current iteration: {}".format(i))
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic=mr.mimic(problem, max_attempts=20, random_state=0, curve=True, keep_pct=i)
    flip_flop_best_states_mimic.append(best_state_mimic)
    flip_flop_best_fitnesses_mimic.append(best_fitness_mimic)
    flip_flop_best_fitness_curves_mimic.append(fitness_curve_mimic)
    

flip_flop_best_fitness_final_mimic=max(flip_flop_best_fitnesses_mimic)
flip_flop_index_best_fitness_mimic=flip_flop_best_fitnesses_mimic.index(flip_flop_best_fitness_final_mimic)
flip_flop_best_state_mimic_final=flip_flop_best_states_mimic[flip_flop_index_best_fitness_mimic]
flip_flop_best_fitness_curve_mimic_final=flip_flop_best_fitness_curves_mimic[flip_flop_index_best_fitness_mimic]

end=time.time()
final_time=end-start
print("Time for MIMIC in Flip Flop ", final_time)

print("Maximum value for MIMIC", flip_flop_best_fitness_final_mimic, " occurs at index " , flip_flop_index_best_fitness_mimic)
print("Best State")
print(flip_flop_best_state_mimic_final)
plt.title("N-Flip Flop Fitness Curve - {}".format("MIMIC"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(flip_flop_best_fitness_curve_mimic_final[:,1], flip_flop_best_fitness_curve_mimic_final[:,0])
plt.show()

plt.title("Fitness Curve - {}".format("All"))
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.plot(flip_flop_best_fitness_curve_rhc_final[:,1], flip_flop_best_fitness_curve_rhc_final[:,0])
plt.plot(flip_flop_best_fitness_curve_sa_final[:,1], flip_flop_best_fitness_curve_sa_final[:,0])
plt.plot(flip_flop_best_fitness_curve_ga_final[:,1], flip_flop_best_fitness_curve_ga_final[:,0])
plt.plot(flip_flop_best_fitness_curve_mimic_final[:,1], flip_flop_best_fitness_curve_mimic_final[:,0])
plt.show()

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(flip_flop_best_fitness_curve_rhc_final[:,1], flip_flop_best_fitness_curve_rhc_final[:,0], color="black", linewidth=1, label='RHC')
ax1.set_ylabel("Fitness")
ax1.set_xlabel("Iteration")
ax1.set_ylim([0, 200])
ax1.legend(loc="upper left")

ax2.plot(flip_flop_best_fitness_curve_sa_final[:,1], flip_flop_best_fitness_curve_sa_final[:,0],color="blue", linewidth=1, label='SA')
ax2.set_ylabel("Fitness")
ax2.set_xlabel("Iteration")
ax2.set_ylim([0, 200])
ax2.legend(loc="upper left")

ax3.plot(flip_flop_best_fitness_curve_ga_final[:,1], flip_flop_best_fitness_curve_ga_final[:,0], color="red", linewidth=1, label='GA')
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Iteration")
ax3.set_ylim([0, 200])
ax3.legend(loc="upper left")

ax4.plot(flip_flop_best_fitness_curve_mimic_final[:,1], flip_flop_best_fitness_curve_mimic_final[:,0], color="orange", linewidth=1, label='MIMIC')
ax4.set_ylabel("Fitness")
ax4.set_xlabel("Iteration")
ax4.set_ylim([0, 200])
ax4.legend(loc="upper left")

figure.suptitle("Comparison Between RHC, SA, GA, MIMIC - Flip Flop Problem")

plt.tight_layout()
plt.show()
############################################################# CHANGING SIZE #############################################################

############################################################# SIX PEAKS PROBLEM #############################################################
fitness_function=mr.SixPeaks(t_pct=0.2)

initial_states_six_peaks=[]

#Create lists to hold the values inside the loops
peaks_best_states_rhc=[]
peaks_best_fitnesses_rhc=[]
peaks_best_fitness_curves_rhc=[]
iterations_rhc=[]
times_rhc=[]

peaks_best_states_sa=[]
peaks_best_fitnesses_sa=[]
peaks_best_fitness_curves_sa=[]
iterations_sa=[]
times_sa=[]

peaks_best_states_ga=[]
peaks_best_fitnesses_ga=[]
peaks_best_fitness_curves_ga=[]
iterations_ga=[]
times_ga=[]

peaks_best_states_mimic=[]
peaks_best_fitnesses_mimic=[]
peaks_best_fitness_curves_mimic=[]
iterations_mimic=[]
times_mimic=[]

for i in range(5,100, 1):
    initial_state=[]
    size=i
    for i in range(0,size,1):
        initial_state.append(random.randint(0,1))
    
    initial_states_six_peaks.append(initial_state)

######RHC
for i in range(5,100, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc=mr.random_hill_climb(problem, max_attempts=20, init_state = initial_states_six_peaks[i-5], random_state=0, curve=True, restarts=50) 
    end_time=time.time()
    final_time=end_time-start_time
    times_rhc.append(final_time)
    iterations_rhc.append(max(fitness_curve_rhc[:,1]))
    peaks_best_fitnesses_rhc.append(best_fitness_rhc)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,100, 1),times_rhc, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,100, 1),iterations_rhc,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,100, 1),peaks_best_fitnesses_rhc, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Random Hill Climbing -  Six Peaks Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######SA
for i in range(5,100, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    schedule=mr.ExpDecay(init_temp=90)
    best_state_sa, best_fitness_sa, fitness_curve_sa= mr.simulated_annealing(problem, max_attempts=20, init_state = initial_states_six_peaks[i-5], schedule = schedule, random_state = 0, curve=True)
    end_time=time.time()
    final_time=end_time-start_time
    times_sa.append(final_time)
    iterations_sa.append(max(fitness_curve_sa[:,1]))
    peaks_best_fitnesses_sa.append(best_fitness_sa)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,100, 1),times_sa, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,100, 1),iterations_sa,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,100, 1),peaks_best_fitnesses_sa, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Simulated Annealing -  Six Peaks Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######GA
for i in range(5,100, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    best_state_ga, best_fitness_ga, fitness_curve_ga=mr.genetic_alg(problem, random_state=0, curve=True, pop_size=800)
    end_time=time.time()
    final_time=end_time-start_time
    times_ga.append(final_time)
    iterations_ga.append(max(fitness_curve_ga[:,1]))
    peaks_best_fitnesses_ga.append(best_fitness_ga)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,100, 1),times_ga, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,100, 1),iterations_ga,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,100, 1),peaks_best_fitnesses_ga, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Genetic Algorithm -  Six Peaks Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######MIMIC
for i in range(5,100, 1):
    
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic=mr.mimic(problem, max_attempts=20, random_state=0, curve=True, keep_pct=.5)
    
    end_time=time.time()
    final_time=end_time-start_time
    print("i: ", i, "time: ", final_time, " Seconds")
    times_mimic.append(final_time)
    iterations_mimic.append(max(fitness_curve_mimic[:,1]))
    peaks_best_fitnesses_mimic.append(best_fitness_mimic)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,100, 1),times_mimic, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,100, 1),iterations_mimic,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,100, 1),peaks_best_fitnesses_mimic, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("MIMC -  Six Peaks Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()


labels = ["RHC","SA", "GA", "MIMIC"]
figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

x1=ax1.plot(range(5,100, 1),times_rhc, color="black", linewidth=.5)[0]
x2=ax1.plot(range(5,100, 1),times_sa, color="blue", linewidth=.5)[0]
x3=ax1.plot(range(5,100, 1),times_ga, color="red", linewidth=.5)[0]
x4=ax1.plot(range(5,100, 1),times_mimic, color="orange", linewidth=.5)[0]
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,100, 1),iterations_rhc,color="black", linewidth=.5)
ax2.plot(range(5,100, 1),iterations_sa,color="blue", linewidth=.5)
ax2.plot(range(5,100, 1),iterations_ga,color="red", linewidth=.5)
ax2.plot(range(5,100, 1),iterations_mimic,color="orange", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,100, 1),peaks_best_fitnesses_rhc, color="black", linewidth=.5)
ax3.plot(range(5,100, 1),peaks_best_fitnesses_sa, color="blue", linewidth=.5)
ax3.plot(range(5,100, 1),peaks_best_fitnesses_ga, color="red", linewidth=.5)
ax3.plot(range(5,100, 1),peaks_best_fitnesses_mimic, color="orange", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Overall Comparison -  Six Peaks Problem")
figure.legend([x1, x2, x3, x4],     
           labels=labels,   
           loc="lower right",   
           borderaxespad=0.1,   
           title="Algorithm"  
           )
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()
############################################################# N Queens #############################################################  

fitness_function = mr.CustomFitness(queens_max)

initial_states_N_queens=[]

#Create lists to hold the values inside the loops
queens_best_states_rhc=[]
queens_best_fitnesses_rhc=[]
queens_best_fitness_curves_rhc=[]
queens_iterations_rhc=[]
queens_itimes_rhc=[]

queens_best_states_sa=[]
queens_best_fitnesses_sa=[]
queens_best_fitness_curves_sa=[]
queens_iterations_sa=[]
queens_itimes_sa=[]

queens_best_states_ga=[]
queens_best_fitnesses_ga=[]
queens_best_fitness_curves_ga=[]
queens_iterations_ga=[]
queens_itimes_ga=[]

queens_best_states_mimic=[]
queens_best_fitnesses_mimic=[]
queens_best_fitness_curves_mimic=[]
queens_iterations_mimic=[]
queens_itimes_mimic=[]

for i in range(5,51, 1):
    initial_state=np.arange(0,i,1)
    initial_states_N_queens.append(initial_state)

######RHC
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=i)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc=mr.random_hill_climb(problem, max_attempts=20, init_state = initial_states_N_queens[i-5], random_state=0, curve=True, restarts=30) 
    end_time=time.time()
    final_time=end_time-start_time
    queens_itimes_rhc.append(final_time)
    queens_iterations_rhc.append(max(fitness_curve_rhc[:,1]))
    queens_best_fitnesses_rhc.append(best_fitness_rhc)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),queens_itimes_rhc, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),queens_iterations_rhc,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),queens_best_fitnesses_rhc, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Random Hill Climbing -  N Queens Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######SA
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=i)
    schedule=mr.ExpDecay(init_temp=185)
    best_state_sa, best_fitness_sa, fitness_curve_sa= mr.simulated_annealing(problem, max_attempts=20, init_state = initial_states_N_queens[i-5], schedule = schedule, random_state = 0, curve=True)
    end_time=time.time()
    final_time=end_time-start_time
    queens_itimes_sa.append(final_time)
    queens_iterations_sa.append(max(fitness_curve_sa[:,1]))
    queens_best_fitnesses_sa.append(best_fitness_sa)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),queens_itimes_sa, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),queens_iterations_sa,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),queens_best_fitnesses_sa, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Simulated Annealing -  N Queens Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######GA
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=i)
    best_state_ga, best_fitness_ga, fitness_curve_ga=mr.genetic_alg(problem, random_state=0, curve=True, pop_size=100)
    end_time=time.time()
    final_time=end_time-start_time
    queens_itimes_ga.append(final_time)
    queens_iterations_ga.append(max(fitness_curve_ga[:,1]))
    queens_best_fitnesses_ga.append(best_fitness_ga)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),queens_itimes_ga, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),queens_iterations_ga,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),queens_best_fitnesses_ga, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Genetic Algorithm -  N Queens Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######MIMIC
for i in range(5,51, 1):
    
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=i)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic=mr.mimic(problem, max_attempts=40, random_state=0, curve=True, keep_pct=.5)
    end_time=time.time()
    final_time=end_time-start_time
    print("i: ", i, "time: ", final_time, " Seconds")
    queens_itimes_mimic.append(final_time)
    queens_iterations_mimic.append(max(fitness_curve_mimic[:,1]))
    queens_best_fitnesses_mimic.append(best_fitness_mimic)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),queens_itimes_mimic, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),queens_iterations_mimic,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),queens_best_fitnesses_mimic, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("MIMC - N Queens Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()


labels = ["RHC","SA", "GA", "MIMIC"]
figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

x1=ax1.plot(range(5,51, 1),queens_itimes_rhc, color="black", linewidth=.5)[0]
x2=ax1.plot(range(5,51, 1),queens_itimes_sa, color="blue", linewidth=.5)[0]
x3=ax1.plot(range(5,51, 1),queens_itimes_ga, color="red", linewidth=.5)[0]
x4=ax1.plot(range(5,51, 1),queens_itimes_mimic, color="orange", linewidth=.5)[0]
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),queens_iterations_rhc,color="black", linewidth=.5)
ax2.plot(range(5,51, 1),queens_iterations_sa,color="blue", linewidth=.5)
ax2.plot(range(5,51, 1),queens_iterations_ga,color="red", linewidth=.5)
ax2.plot(range(5,51, 1),queens_iterations_mimic,color="orange", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),queens_best_fitnesses_rhc, color="black", linewidth=.5)
ax3.plot(range(5,51, 1),queens_best_fitnesses_sa, color="blue", linewidth=.5)
ax3.plot(range(5,51, 1),queens_best_fitnesses_ga, color="red", linewidth=.5)
ax3.plot(range(5,51, 1),queens_best_fitnesses_mimic, color="orange", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Overall Comparison - N Queens Problem")
figure.legend([x1, x2, x3, x4],     
           labels=labels,   
           loc="lower right",   
           borderaxespad=0.1,   
           title="Algorithm"  
           )
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

############################################################# Flip Flops #############################################################  

fitness_function = fitness_function=mr.FlipFlop()

initial_states_N_flipflop=[]

#Create lists to hold the values inside the loops
flipflop_best_states_rhc=[]
flipflop_best_fitnesses_rhc=[]
flipflop_best_fitness_curves_rhc=[]
flipflop_iterations_rhc=[]
flipflop_itimes_rhc=[]

flipflop_best_states_sa=[]
flipflop_best_fitnesses_sa=[]
flipflop_best_fitness_curves_sa=[]
flipflop_iterations_sa=[]
flipflop_itimes_sa=[]

flipflop_best_states_ga=[]
flipflop_best_fitnesses_ga=[]
flipflop_best_fitness_curves_ga=[]
flipflop_iterations_ga=[]
flipflop_itimes_ga=[]

flipflop_best_states_mimic=[]
flipflop_best_fitnesses_mimic=[]
flipflop_best_fitness_curves_mimic=[]
flipflop_iterations_mimic=[]
flipflop_itimes_mimic=[]

for i in range(5,51, 1):
    initial_state=[]
    size=i
    for i in range(0,size,1):
        initial_state.append(random.randint(0,1))
    
    initial_states_N_flipflop.append(initial_state)

######RHC
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc=mr.random_hill_climb(problem, max_attempts=20, init_state = initial_states_N_flipflop[i-5], random_state=0, curve=True, restarts=50) 
    end_time=time.time()
    final_time=end_time-start_time
    flipflop_itimes_rhc.append(final_time)
    flipflop_iterations_rhc.append(max(fitness_curve_rhc[:,1]))
    flipflop_best_fitnesses_rhc.append(best_fitness_rhc)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),flipflop_itimes_rhc, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),flipflop_iterations_rhc,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),flipflop_best_fitnesses_rhc, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Random Hill Climbing -  Flip Flop Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######SA
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    schedule=mr.ExpDecay(init_temp=60)
    best_state_sa, best_fitness_sa, fitness_curve_sa= mr.simulated_annealing(problem, max_attempts=20, init_state = initial_states_N_flipflop[i-5], schedule = schedule, random_state = 0, curve=True)
    end_time=time.time()
    final_time=end_time-start_time
    flipflop_itimes_sa.append(final_time)
    flipflop_iterations_sa.append(max(fitness_curve_sa[:,1]))
    flipflop_best_fitnesses_sa.append(best_fitness_sa)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),flipflop_itimes_sa, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),flipflop_iterations_sa,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),flipflop_best_fitnesses_sa, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Simulated Annealing -  Flip Flop Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######GA
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    best_state_ga, best_fitness_ga, fitness_curve_ga=mr.genetic_alg(problem, random_state=0, curve=True, pop_size=300)
    end_time=time.time()
    final_time=end_time-start_time
    flipflop_itimes_ga.append(final_time)
    flipflop_iterations_ga.append(max(fitness_curve_ga[:,1]))
    flipflop_best_fitnesses_ga.append(best_fitness_ga)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),flipflop_itimes_ga, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),flipflop_iterations_ga,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),flipflop_best_fitnesses_ga, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Genetic Algorithm -  Flip Flop Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

######MIMIC
for i in range(5,51, 1):
    start_time=time.time()
    problem=mr.DiscreteOpt(length=i, fitness_fn=fitness_function, maximize=True, max_val=2)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic=mr.mimic(problem, max_attempts=10, random_state=0, curve=True, keep_pct=.25)
    end_time=time.time()
    final_time=end_time-start_time
    print("i: ", i, "time: ", final_time, " Seconds")
    flipflop_itimes_mimic.append(final_time)
    flipflop_iterations_mimic.append(max(fitness_curve_mimic[:,1]))
    flipflop_best_fitnesses_mimic.append(best_fitness_mimic)

figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

ax1.plot(range(5,51, 1),flipflop_itimes_mimic, color="black", linewidth=.5)
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),flipflop_iterations_mimic,color="black", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),flipflop_best_fitnesses_mimic, color="black", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("MIMC - Flip Flop Problem")
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()


labels = ["RHC","SA", "GA", "MIMIC"]
figure, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

x1=ax1.plot(range(5,51, 1),flipflop_itimes_rhc, color="black", linewidth=.5)[0]
x2=ax1.plot(range(5,51, 1),flipflop_itimes_sa, color="blue", linewidth=.5)[0]
x3=ax1.plot(range(5,51, 1),flipflop_itimes_ga, color="red", linewidth=.5)[0]
x4=ax1.plot(range(5,51, 1),flipflop_itimes_mimic, color="orange", linewidth=.5)[0]
ax1.set_ylabel("Time (sec)")
ax1.set_xlabel("Problem Size")

ax2.plot(range(5,51, 1),flipflop_iterations_rhc,color="black", linewidth=.5)
ax2.plot(range(5,51, 1),flipflop_iterations_sa,color="blue", linewidth=.5)
ax2.plot(range(5,51, 1),flipflop_iterations_ga,color="red", linewidth=.5)
ax2.plot(range(5,51, 1),flipflop_iterations_mimic,color="orange", linewidth=.5)
ax2.set_ylabel("Iterations")
ax2.set_xlabel("Problem Size")

ax3.plot(range(5,51, 1),flipflop_best_fitnesses_rhc, color="black", linewidth=.5)
ax3.plot(range(5,51, 1),flipflop_best_fitnesses_sa, color="blue", linewidth=.5)
ax3.plot(range(5,51, 1),flipflop_best_fitnesses_ga, color="red", linewidth=.5)
ax3.plot(range(5,51, 1),flipflop_best_fitnesses_mimic, color="orange", linewidth=.5)
ax3.set_ylabel("Fitness")
ax3.set_xlabel("Problem Size")
figure.suptitle("Overall Comparison - Flip Flop Problem")
figure.legend([x1, x2, x3, x4],     
           labels=labels,   
           loc="lower right",   
           borderaxespad=0.1,   
           title="Algorithm"  
           )
ax4.set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()

#############################################################################################

#Helpful Functions
######################################################
#Function to plot learning curves
#Source: Scikit Learn webpage.
#Article "Plotting Learning Curves"
#Code retrieved from 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.0, 1.01), cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    print("Train score means: " + str(train_scores_mean))
    print("Test score means: " + str(test_scores_mean))
    print("")
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fitting Times (seconds)")
    axes[1].set_title("Scalability of the Model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fitting Times (seconds)")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the Model")

    return plt

#Create two random sets for classification
full_1_X, full_1_y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, random_state=seed, class_sep=.1, shuffle=True)
full_2_X, full_2_y = make_classification(n_samples=8000, n_features=10, n_informative=2, n_redundant=1, n_clusters_per_class=2, random_state=seed, flip_y=.3, class_sep=.5, shuffle=True, weights=[0.7,0.3])

#Scale the data for easier processing by algorithm
min_max_scaler =  MinMaxScaler()
full_1_X_minmax = min_max_scaler.fit_transform(full_1_X)
full_2_X_minmax = min_max_scaler.fit_transform(full_2_X)

#Show interesting features of dataset 1
full_1=pd.DataFrame(full_1_X_minmax)
full_1['y']=full_1_y
check_data1=full_1.describe().transpose()
full_1['y'].describe()
plt.show()

#Show interesting features of dataset 2
full_2=pd.DataFrame(full_2_X_minmax)
full_2['y']=full_2_y
#sn.heatmap(full_2.corr(), annot=True)
check_data2=full_2.describe().transpose()
full_2['y'].describe()
plt.show()


#Create a training dataset and a holdout set. The training set will be 
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(full_1_X_minmax, full_1_y, test_size=0.3, random_state=seed)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(full_2_X_minmax, full_2_y, test_size=0.3, random_state=seed)

#################################################RHC NEURAL NETWORK###############################################
#Code to run random optimization algorithms for neural networks weight
#Source: mlrose documentation webpage.
#Article "Plotting Learning Curves"
#Code retrieved from 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

start=time.time()
original = MLPClassifier(activation='tanh', hidden_layer_sizes=[5,5,5], max_iter=2000, solver='adam', learning_rate_init=.001,random_state=seed)
NN=original.fit(X_1_train, y_1_train)
end=time.time()
final_time=end-start

pred_train=original.predict(X_1_train)
pred_test=original.predict(X_1_test)

print("Original Neural Network - Dataset 1")
print("The training accuracy of the original neural network on training dataset 1 is: ", accuracy_score(y_1_train, pred_train))
print("The test accuracy of the original neural network on training dataset 1 is: ", accuracy_score(y_1_test, pred_test))
print("The network took: ", final_time, " seconds to train.")


#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Training Dataset 1 Original")

plot_conf = plot_confusion_matrix(original, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Training Dataset 1 Original")

plt.show()


ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Testing Dataset 1 Original")

plot_conf = plot_confusion_matrix(original, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Testing Dataset 1 Original")

plt.show()

################################################# RHC NEURAL NETWORK ######################################################
start=time.time()
nn_rhc = mr.NeuralNetwork(hidden_nodes = [5,5,5], activation = 'tanh', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 1, max_attempts = 50, \
                                 random_state = 0, restarts=50, curve=True)

nn_rhc.fit(X_1_train, y_1_train)
end=time.time()
final_time=end-start

pred_train=nn_rhc.predict(X_1_train)
pred_test=nn_rhc.predict(X_1_test)

print("RHC Neural Network - Dataset 1")
print("The training accuracy of the RHC neural network on training dataset 1 is: ", accuracy_score(y_1_train, pred_train))
print("The test accuracy of the RHC neural network on training dataset 1 is: ", accuracy_score(y_1_test, pred_test))
print("The network took: ", final_time, " seconds to train.")
print("")

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Training Dataset 1 RHC")

plot_conf = plot_confusion_matrix(nn_rhc, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Training Dataset 1 RHC")

plt.show()


ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Testing Dataset 1 RHC")

plot_conf = plot_confusion_matrix(nn_rhc, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Testing Dataset 1 RHC")

plt.show()


################################################# SA NEURAL NETWORK ######################################################
start=time.time()
nn_sa = mr.NeuralNetwork(hidden_nodes = [5,5,5], activation = 'tanh', \
                                 algorithm = 'simulated_annealing', max_iters = 50000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 1, max_attempts = 50, \
                                 random_state = 0, schedule=mr.ExpDecay(init_temp=500), curve=True)

NN_SA=nn_sa.fit(X_1_train, y_1_train)
end=time.time()
final_time=end-start

pred_train=nn_sa.predict(X_1_train)
pred_test=nn_sa.predict(X_1_test)

print("SA Neural Network - Dataset 1")
print("The training accuracy of the SA neural network on training dataset 1 is: ", accuracy_score(y_1_train, pred_train))
print("The test accuracy of the SA neural network on training dataset 1 is: ", accuracy_score(y_1_test, pred_test))
print("The network took: ", final_time, " seconds to train.")
print("")

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Training Dataset 1 SA")

plot_conf = plot_confusion_matrix(nn_sa, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Training Dataset 1 SA")

plt.show()


ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Testing Dataset 1 SA")

plot_conf = plot_confusion_matrix(nn_sa, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Testing Dataset 1 SA")

plt.show()


################################################# GA NEURAL NETWORK ######################################################
start=time.time()
nn_ga = mr.NeuralNetwork(hidden_nodes = [5,5,5], activation = 'tanh', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 1, max_attempts = 50, \
                                 random_state = 0, pop_size=500, curve=True)

nn_ga.fit(X_1_train, y_1_train)
end=time.time()
final_time=end-start


pred_train=nn_ga.predict(X_1_train)
pred_test=nn_ga.predict(X_1_test)

print("GA Neural Network - Dataset 1")
print("The training accuracy of the GA neural network on training dataset 1 is: ", accuracy_score(y_1_train, pred_train))
print("The test accuracy of the GA neural network on training dataset 1 is: ", accuracy_score(y_1_test, pred_test))
print("The network took: ", final_time, " seconds to train.")
print("")

#Find the error on the full training dataset
#Code to plot classification report
#Source: Adapted from Stack Overflow example by user akilat90
#Article "How to plot scikit learn classification report?"
#Code modified from 
#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

ax = plt.axes()
clf_report = classification_report(y_1_train,
                                   pred_train,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Training Dataset 1 GA")

plot_conf = plot_confusion_matrix(nn_ga, X_1_train, y_1_train,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Training Dataset 1 GA")

plt.show()


ax = plt.axes()
clf_report = classification_report(y_1_test,
                                   pred_test,
                                   output_dict=True)
sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
ax.set_title("Classification Report - Neural Networks - Testing Dataset 1 GA")

plot_conf = plot_confusion_matrix(nn_ga, X_1_test, y_1_test,
                                 cmap=plt.cm.Blues, normalize='true')
plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - Testing Dataset 1 GA")

plt.show()



#from sklearn.model_selection import cross_val_score
#cross_val_score

start=time.time()
percentage = np.arange (.2, 1.2, .2)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
plot_learning_curve(original, "Learning Curves: Original - Dataset 1", X_1_train, y_1_train, axes=axes, ylim=(0.0, 1.01), cv=5, n_jobs=-1, train_sizes=percentage)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()
end=time.time()
final=end-start
print("Original Neural Network took ", final , " Seconds to train.")

start=time.time()
percentage = np.arange (.2, 1.2, .2)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
plot_learning_curve(nn_rhc, "Learning Curves: Random Hill Climbing - Dataset 1", X_1_train, y_1_train, axes=axes, ylim=(0.0, 1.01), cv=5, n_jobs=-1, train_sizes=percentage)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()
end=time.time()
final=end-start
print("RHC Neural Network took ", final , " Seconds to train.")

start=time.time()
percentage = np.arange (.2, 1.0, .2)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
plot_learning_curve(nn_sa, "Learning Curves: Simulated Annealing - Dataset 1", X_1_train, y_1_train, axes=axes, ylim=(0.0, 1.01), cv=5, n_jobs=-1, train_sizes=percentage)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()
end=time.time()
final=end-start
print("SA Neural Network took ", final , " Seconds to train.")

start=time.time()
percentage = np.arange (.2, 1.0, .2)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
plot_learning_curve(nn_sa, "Learning Curves: Genetic Algorithm - Dataset 1", X_1_train, y_1_train, axes=axes, ylim=(0.0, 1.01), cv=5, n_jobs=-1, train_sizes=percentage)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.show()
end=time.time()
final=end-start
print("GA Neural Network took ", final , " Seconds to train.")