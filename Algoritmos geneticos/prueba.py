import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import differential_evolution

min_ = -4 
max_ = 4 
bounds = [[min_,max_],[min_,max_]]
seed = 127
dim = 4

# Confidence level etc
alpha = 0.05
beta = 1. - alpha
critical_chi_sq = chi2.isf(alpha, 2)
critical_loglike = 0.5 * critical_chi_sq
min_chi_sq = 0.

# Color style for output sample points
de_pts = "#91bfdb" # Diver scan
rn_pts = "#fc8d59" # Random scan
gd_pts = "#ffffbf" # Grid scan

def func(p):
	x,y = p 
	r = np.sqrt(x**2 + y**2)
	return np.sqrt(r) 


def rosenbrock(x, y):
    """
    @returns Rosenbrock function
    """
    a = 1.
    b = 100.
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_general(x):
    """
    @returns Generalization of Rosenbrock function
    """
    n = len(x)
    return sum(rosenbrock(x[i], x[i+1]) for i in range(n - 1))

def loglike(x):
    """
    @returns Log-likelihood
    """
    return -rosenbrock_general(x)

def samples_inside(x,chi_sq): 
    delta_chi_sq = chi_sq - min_chi_sq
    inside = delta_chi_sq <= critical_chi_sq
    return x[:, inside]

def de_scan(dim,round_to_nearest=None): 
	x = [] 
	chi_sq = [] 

	def objective(x_): 
		chi_sq_ = loglike(x_)
		chi_sq.append(chi_sq_)
		x.append(x_)
		return chi_sq_

	differential_evolution(objective, bounds,
                           strategy='rand1bin', maxiter=None,
                           popsize=50, tol=0.01, mutation=(0.7, 1.99999), recombination=0.15,
                           polish=False, seed=seed)

	if round_to_nearest is not None: 
		len_x=len(x) 
		keep_n = len_x - (len_x %round_to_nearest) 
		x = x[:keep_n]
		chi_sq = chi_sq[:keep_n]

	return samples_inside(np.array(x).T, np.array(chi_sq)), len(x)

if __name__ == '__main__': 
    np.random.seed(seed) 
    plt.figure(figsize=(10,10))
    print("Running de_scan") 
    t0 = time.time()
    x , calls = de_scan(dim,round_to_nearest = 1000) 

    de_time = time.time() - t0
    print("Plotting de_scan") 
    print(r'tiempo de ejecución: '+str(de_time))
    plt.scatter(x[0], x[1], s=30, edgecolor='0.05',
                linewidth=0.25, alpha=1.0, facecolor=de_pts,
                label=r'Differential evolution: \num{{{:d}}} out of \num{{{:d}}} points'
                .format(x.shape[1], calls))
    
    #print(de_scatter)
    plt.title("Función de Rosenbrock")
    plt.xlabel("parametro X1") 
    plt.ylabel("Parametro X2")
    plt.savefig("Grafico_Rosenbrock.svg")
    plt.show()