import numpy as np 
import subprocess
from scipy.optimize import differential_evolution
from scipy.stats import chi2 
from scipy.interpolate import interp1d #Para interpolar los datos experimentales.
import matplotlib.pyplot as plt 
import pandas as pd 
import time
import random

#Ligaduras del sistema. 
lash_min = -4
lash_max = 0
mass_min = 1 # Iniciamos en una masa de 10 GeV
mass_max = 4 # Finalizamos en una masa de 10000 GeV
mu_min = 0
mu_max = 4
#seed = 16
#np.random.seed(seed)
seed = random.randint(1, 1000)

# Configura la semilla aleatoria para la generación de números aleatorios
random.seed(seed)

print("La semilla es:",seed)

sltns = 0
min_chi_sq = 0.
alpha = 0.05
critical_chi_sq = chi2.isf(alpha,2)

LZ22=np.loadtxt('Lz_experiment.txt')   #Tener presente que solo funciona a partir de 9 GeV
fLZ22 = interp1d(LZ22[:,0],LZ22[:,5]*1e36) #Función de datos experimentales interpolada.

def filter(df_,alpha = 0.05,min_chi_sq=0.): 
	#min_chi_sq = 0.
	critical_chi_sq = chi2.isf(alpha,2)
	df_['chi'] = df_['chi'] - min_chi_sq
	filtro = df_['chi'] <= critical_chi_sq #Filtra los datos a una valor determina de chi cuadrado
	df_ = df_[filtro]
	df_ = df_.reset_index() #Resetea los indices. 
	return df_

def writer(file,dictionary):
	data1=open(file,'w')
	for items in dictionary.items(): 
		data1.write("%s %s\n"%items)
	data1.close()

def samples_inside(x,chi_sq): 
    delta_chi_sq = chi_sq - min_chi_sq
    inside = delta_chi_sq <= critical_chi_sq
    return x[inside, :]

def parametros(X):
	#---------------- Rutas de los archivos -------------------------------
	ruta = 'data.dat' #Ruta para guardar el archivo.
	rutaG = './main data.dat > temporal.dat' #Ruta para ejecutar micromegas 
	#----------------------------------------------------------------------
	#------Diccionario con los datos del archivo. -------------------------
	data = {'Mh':125, 'laphi':0.07,'laSH1':0.1,'Mp1':300,'mu32':1000,'Mtop':173.2} 
	data['Mp1'] = 10**X[1] #Valores de la masa de la particulas.
	data['laSH1'] = 10**X[0] #Valores de laSH
	data['mu32'] = 10**X[2] #Valores de mu3
	#----------------------------------------------------------------------
	#--------------------Sistema de escritura en el archivo----------------
	writer(ruta,data) 
	#Corriendo micromegas y extrayendo la densidad reliquia 
	subprocess.getoutput(rutaG)

def omega(): 
	omeg = 0.0 
	COMMAND = "grep 'Omega' temporal.dat | awk 'BEGIN{FS=\"=\"};{print $3}'"  
	dato = subprocess.getoutput(COMMAND) #ejecutar el comando desde la terminal 
	if (len(dato)>0): 
		omeg = float(dato) 
	else: 
		omeg = -1 
	return omeg

def csection(): 
	cs = 0.0
	COMMAND = "grep 'proton  SI' temporal.dat | awk '{print $3}'"
	dato = subprocess.getoutput(COMMAND)
	if(len(dato)>0):
		cs = float(dato)
	else: 
		cs = -1 
	return cs

def gaussian(X):
	parametros(X)
	global fLZ22
	x_densidad = omega() #Calcula la densidad reliquia. 
	x_directa = csection() #Calcula la cross section independiente de espin.
	
	sigma_obs_densidad = 0.001 #Error del valor observado para la densidad reliquia.
	sigma_obs_cross_section = fLZ22(10**X[1])/1.64 #Error del valor observado en el experimento LZ.
	print("Los valores de sigma experimentales")
	print(sigma_obs_densidad,sigma_obs_cross_section)
	sigma_the_densidad = 0.1*x_densidad #Error del valor teorico para la densidad reliquia
	sigma_the_cross_section = 0.2*x_directa #Error del valor teorico para la detección directa.
	print("los valores de sigma teoricos")
	print(sigma_the_densidad,sigma_the_cross_section)
	sigma_densidad = sigma_the_densidad**2 + sigma_obs_densidad**2 #Sigma de la densidad reliquia.
	sigma_directa = sigma_the_cross_section**2 + sigma_obs_cross_section**2 #Sigma de la cross section.
	print("Los valores de sigma")
	print(sigma_densidad,sigma_directa)
	xobs_densidad = 0.120 #densidad reliquia observable.
	xobs_cross_section = 0 #cross section observada (se establece en 0)

	val_densidad= ((x_densidad-xobs_densidad)**2)/sigma_densidad #loglikelihood para la densidad reliquia
	val_directa = ((x_directa-xobs_cross_section)**2)/sigma_directa #loglikelihood para la cross section
	return (val_densidad + val_directa),x_directa


def de_scan(round_to_nearest=None):

    x = []
    chi_sq = []

    bounds = [(lash_min, lash_max), (mass_min, mass_max), (mu_min, mu_max)]  # ligaduras

    def objective(x_):
    	arreglo = [0]*5
    	chi_sq_, cross_section = gaussian(x_)

    	chi_sq.append(chi_sq_)

    	arreglo[0] = x_[0]
    	arreglo[1] = x_[1]
    	arreglo[2] = x_[2] 
    	arreglo[3] = cross_section
    	arreglo[4] = chi_sq_
    	
    	x.append(arreglo)

    	if (len(x) % 1000 == 0):
    		print(len(x),end='\r')
    	return chi_sq_

    differential_evolution(objective, bounds,
                           strategy='randtobest1exp', maxiter=None,
                           popsize=200, tol=0.01, mutation=(1.0, 1.999), recombination=0.9,
                           polish=False, seed=seed)
    #currenttobest1exp -> archivo_profile_2.csv
    #best1bin -> archivo_profile.csv
    #best2exp -> archivo_profile_1.csv
    #rand1bin -> archivo_profile_rand1bin.csv
    #randtobest1exp -> archivo_profile_randtobest1exp.csv

    try:
    	try:
    		column_names = ['laSH', 'mass', 'mu','cross_section','chi']
    		df = pd.DataFrame(np.array(x), columns=column_names)
    		print(df.head())
    	except:
    		df = pd.DataFrame(np.array(x))

    	try:
    		df = filter(df)
    		df.to_csv('archivo_profile_randtobest1exp.csv', index=False, header=None)
    		print("Datos almacenados con filtro")
    		print("El tamaño de los datos es:",len(df))

    	except:
    		df.to_csv('archivo_profile_randtobest1exp.csv', index=False, header=None)
    		print("Datos almacenados sin filtro")

    	print("Datos almacenados con exito")
    except:
    	print("Los datos del dataframe no han podido ser almacenados")

    return samples_inside(np.array(x),np.array(chi_sq)),len(x)


if __name__ == '__main__':
	
	seed = 16
	np.random.seed(seed) 
	print("Running de_scan") 
	tO = time.time()
	x,call = de_scan()
	de_time = time.time() - tO 
	de_time = de_time/60
	print("Tiempo de ejecución: ", de_time, " minutos")
	#print("Cantidad de datos generados: ", call)
	print("Finalizado")