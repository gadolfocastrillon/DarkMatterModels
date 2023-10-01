import numpy as np 
from scipy.optimize import rosen, differential_evolution 
import matplotlib.pyplot as plt 
from scipy.stats import chi2 
import pandas as pd
import time

#min_ = -5
#max_ = 5 
#dim = 4
#bounds = [(min_,max_)] *(dim)
seed = 127

alpha = 0.05
beta = 1. - alpha 
critical_chi_sq = chi2.isf(alpha,2) 
#print(critical_chi_sq)
min_chi_sq = 0. 

# Color style for output sample points
de_pts = "#91bfdb" # Diver scan
rn_pts = "#fc8d59" # Random scan
gd_pts = "#ffffbf" # Grid scan

np.random.seed(seed) 

def rosenbrock(x,y): 
    a = 1. 
    b = 100.
    return (a-x)**2 + b * (y - x**2)**2

def rosenbrock_general(x): 
    n = len(x) 
    return sum(rosenbrock(x[i],x[i+1]) for i in range(n-1))

def loglike(x): 
    return - rosenbrock_general(x)

def samples_inside(x,chi_sq): 
    delta_chi_sq = chi_sq - min_chi_sq
    inside = delta_chi_sq <= critical_chi_sq
    return x[:, inside]

def filter(df_,alpha = 0.05): 
    min_chi_sq = 0.
    critical_chi_sq = chi2.isf(alpha,2)
    df_['chi'] = df_['chi'] - min_chi_sq
    filtro = df_['chi'] <= critical_chi_sq #Filtra los datos a una valor determina de chi cuadrado
    df_ = df_[filtro]
    df_ = df_.reset_index() #Resetea los indices. 
    return df_

def de_scan(dim,round_to_nearest=None,estrategia = 'rand1bin',nombre_archivo='archivo_profile.csv',min_=-5,max_=5):
    
    bounds = [(min_,max_)] *(dim) 
    x = [] 
    chi_sq = [] 
    x_chi = [] 
    #dim nos dice de cuantos puntos sumo 
    #Activamos la función de Rosenbrock y 
    #guardamos los datos en x y chi_sq 
    def objective(x_): 
        #print(len(x_))
        x_chi_ = [0]*5
        x_chi_[0] = x_[0]
        x_chi_[1] = x_[1]
        x_chi_[2] = x_[2] 
        x_chi_[3] = x_[3]
        
        #chi_sq_ = 2.*rosen(x_) 
        chi_sq_ = -2.*loglike(x_) 
        x_chi_[4] = chi_sq_
        chi_sq.append(chi_sq_) #Guarda los datos desde que se ejecuta differential_evolution
        x.append(x_) 
        x_chi.append(x_chi_)
        return chi_sq_

    #ejecuta objetivo que me guarda el valor de x y chi 
    #y ademas me aplica la funcion de Rosenbrock 
    differential_evolution(objective, bounds,
                           strategy=estrategia, maxiter=None,
                           popsize=50, tol=0.01, mutation=(0.7, 1.99999), recombination=0.7,
                           polish=False, seed=seed)

    column_names = ['x1', 'x2', 'x3','x4','chi']
    df = pd.DataFrame(np.array(x_chi), columns=column_names)
    df = filter(df)
    df.to_csv(nombre_archivo, index=False, header=None)
    print("Datos almacenados con filtro")

    if round_to_nearest is not None: 
        len_x = len(x)
        print(len(x)%round_to_nearest) 
        keep_n = len_x - (len_x %round_to_nearest)
        x = x[:keep_n]
        chi_sq = chi_sq[:keep_n]
    return samples_inside(np.array(x).T, np.array(chi_sq)),len(x)

'''
result = differential_evolution(rosen,bounds,
						strategy='rand1bin', maxiter=None,
                        popsize=50, tol=0.01, mutation=(0.7, 1.99999), recombination=0.15,
                        polish=False, seed=seed) 
'''
def graficador(x,y):
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, s=30, edgecolor='0.05',
                linewidth=0.25, alpha=1.0, facecolor=de_pts,
                label=r'Differential evolution: \num{{{:d}}} out of \num{{{:d}}} points')
    plt.title("Función de Rosenbrock")
    plt.xlabel("parametro X1") 
    plt.ylabel("Parametro X2")
    plt.savefig("Grafico_Rosenbrock.svg")
    plt.show()


if __name__ == '__main__': 
    
    plt.figure(figsize=(10,10))
    print("Running de_scan") 
    t0 = time.time()
    x , calls = de_scan(dim,round_to_nearest = 1000) 
    print(x)
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