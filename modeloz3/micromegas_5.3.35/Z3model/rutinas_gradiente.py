import numpy as np
import subprocess
from scipy.optimize import differential_evolution
from scipy.stats import chi2
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import time
import random

lash_min = -4
lash_max = 0
mass_min = 1
mass_max = 4
mu_min = 0
mu_max = 4
seed = random.randint(1, 1000)
random.seed(seed)

print("La semilla es:", seed)

min_chi_sq = 0.0
alpha = 0.05
critical_chi_sq = chi2.isf(alpha, 2)

LZ22 = np.loadtxt('Lz_experiment.txt')
fLZ22 = interp1d(LZ22[:, 0], LZ22[:, 5] * 1e36)

def filter(df_, alpha=0.05, min_chi_sq=0.0):
    critical_chi_sq = chi2.isf(alpha, 2)
    df_['chi'] = df_['chi'] - min_chi_sq
    filtro = df_['chi'] <= critical_chi_sq
    df_ = df_[filtro]
    df_ = df_.reset_index()
    return df_

def writer(file, dictionary):
    data1 = open(file, 'w')
    for items in dictionary.items():
        data1.write("%s %s\n" % items)
    data1.close()

def samples_inside(x, chi_sq):
    delta_chi_sq = chi_sq - min_chi_sq
    inside = delta_chi_sq <= critical_chi_sq
    return x[inside, :]

def parametros(X):
    ruta = 'data.dat'
    rutaG = './main data.dat > temporal.dat'
    data = {'Mh': 125, 'laphi': 0.07, 'laSH1': 0.1, 'Mp1': 300, 'mu32': 1000, 'Mtop': 173.2}
    data['Mp1'] = 10 ** X[1]
    data['laSH1'] = 10 ** X[0]
    data['mu32'] = 10 ** X[2]
    writer(ruta, data)
    subprocess.getoutput(rutaG)

def omega():
    omeg = 0.0
    COMMAND = "grep 'Omega' temporal.dat | awk 'BEGIN{FS=\"=\"};{print $3}'"
    dato = subprocess.getoutput(COMMAND)
    if len(dato) > 0:
        omeg = float(dato)
    else:
        omeg = -1
    return omeg

def csection():
    cs = 0.0
    COMMAND = "grep 'proton  SI' temporal.dat | awk '{print $3}'"
    dato = subprocess.getoutput(COMMAND)
    if len(dato) > 0:
        cs = float(dato)
    else:
        cs = -1
    return cs

def gradient_gaussian(X):
    parametros(X)
    x_densidad = omega()
    x_directa = csection()

    sigma_obs_densidad = 0.001
    sigma_obs_cross_section = fLZ22(10 ** X[1]) / 1.64

    sigma_the_densidad = 0.1 * x_densidad
    sigma_the_cross_section = 0.2 * x_directa

    sigma_densidad = sigma_the_densidad ** 2 + sigma_obs_densidad ** 2
    sigma_directa = sigma_the_cross_section ** 2 + sigma_obs_cross_section ** 2
    
    grad_densidad = 2 * (x_densidad - 0.120) / sigma_densidad
    grad_directa = 2 * (x_directa - 0) / sigma_directa

    return np.array([grad_densidad, grad_directa])

def gd_optimize(X, learning_rate=0.1, max_iterations=1000, tolerance=1e-5):
    for i in range(max_iterations):
        gradient = gradient_gaussian(X)
        X -= learning_rate * gradient
        if np.linalg.norm(gradient) < tolerance:
            break
    return X

def gd_scan(round_to_nearest=None):
    x = []
    chi_sq = []

    bounds = [(lash_min, lash_max), (mass_min, mass_max), (mu_min, mu_max)]

    def objective(x_):
        arreglo = [0] * 5
        x_ = gd_optimize(x_)
        #chi_sq_, cross_section= gaussian(x_)
        arr = gradient_gaussian(x_)
        
        chi_sq.append(chi_sq_)

        arreglo[0] = x_[0]
        arreglo[1] = x_[1]
        arreglo[2] = x_[2]
        arreglo[3] = cross_section
        arreglo[4] = chi_sq_

        x.append(arreglo)

        if len(x) % 1000 == 0:
            print(len(x), end='\r')
        return chi_sq_

    differential_evolution(objective, bounds,
                           strategy='rand1bin', maxiter=None,
                           popsize=50, tol=0.01, mutation=(1.0, 1.999), recombination=0.9,
                           polish=False, seed=seed)

    try:
        try:
            column_names = ['laSH', 'mass', 'mu', 'cross_section', 'chi']
            df = pd.DataFrame(np.array(x), columns=column_names)
            print(df.head())
        except:
            df = pd.DataFrame(np.array(x))

        try:
            df = filter(df)
            df.to_csv('archivo_profile_rand1bin.csv', index=False, header=None)
            print("Datos almacenados con filtro")
            print("El tamaño de los datos es:", len(df))
        except:
            df.to_csv('archivo_profile_rand1bin.csv', index=False, header=None)
            print("Datos almacenados sin filtro")

        print("Datos almacenados con éxito")
    except:
        print("Los datos del dataframe no han podido ser almacenados")

    return samples_inside(np.array(x), np.array(chi_sq)), len(x)

if __name__ == '__main__':
    seed = 16
    np.random.seed(seed)
    print("Running gd_scan")
    tO = time.time()
    x, call = gd_scan()
    de_time = time.time() - tO
    de_time = de_time / 60
    print("Tiempo de ejecución: ", de_time, " minutos")
    print("Finalizado")