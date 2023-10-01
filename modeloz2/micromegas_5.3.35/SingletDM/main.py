import rutinas as r 
import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.optimize import differential_evolution 
from scipy.stats import chi2

seed = 10
min_chi_sq = 0.
alpha = 0.05
critical_chi_sq = chi2.isf(alpha,2)

mass_ini = 0
mass_fin = 4
laSH_ini = -4
laSH_fin = 0
N = 100000

def de_scan(): 
	chi_sq = [] 
	x = [] 
	bounds =[(laSH_ini,laSH_fin),(mass_ini,mass_fin)]
	def objective(x_):
		chi_sq_,exp = r.gaussian(x_)
		chi_sq.append(chi_sq_)
		val_cross = r.calc_cross_section()
		x.append([x_[0],x_[1],val_cross,chi_sq_,exp])
		r.printf_(x)
		if (len(x) % 1000 == 0):
			print(len(x),end='\r')
		return chi_sq_

	differential_evolution(objective, bounds,
                           strategy='rand1bin', maxiter=None,
                           popsize=100, tol=0.01, mutation=(1.5, 1.999), recombination=0.9,
                           polish=False, seed=seed)
	
	return x,len(x)

def gen_random(mostrar=False): 
	x = [0]*(2)
	x[1] = 10**np.random.uniform(mass_ini,mass_fin)
	x[0] = 10**np.random.uniform(laSH_ini,laSH_fin)
	r.mod_archivo(x)
	val_densidad = r.calc_densidad_reliquia()
	val_cross = r.calc_cross_section()
	
	if(mostrar): 
		print("La densidad reliquia es: ",val_densidad)
		print("La sección transversal es: ",val_cross)

	return [x[0],x[1],val_cross,val_densidad],val_densidad

def aleatorio():
	import time 
	datos = []
	print("Running de_scan")
	t0 = time.time()
	for i in range(N): 
		vec,den = gen_random()
		if ((den<=0.13) and (den>=0.11)):
			datos.append(vec)
			#print(vec)
		if(i %1000 == 0):
			print(i,end='\r')

	print(r'Tiempo de ejecución : '+ str(round(time.time() - t0,2))+ ' segundos') #Imprime el tiempo que demora en ejecutar el progama.
	print(r'Tamaño de los datos obtenidos : ',len(datos))
	print("Finalizado")
	df = pd.DataFrame(np.array(datos))
	df.to_csv('datos_aleatorios_2.csv')
	return datos

def algoritmo_G():
	import time
	np.random.seed(seed)
	print("Running de_scan")
	t0 = time.time()
	vec, calls = de_scan() 
	dt = time.time() - t0
	print(r'Tiempo de ejecución : '+ str(round(dt/60,2))+ ' minutos') #Imprime el tiempo que demora en ejecutar el progama.
	print(r'Tamaño de los datos obtenidos : ',calls)
	print("Finalizado")
	#Crea un dataframe y ademas filtra los datos de interes.
	nombres = ['lash','mass','cross_section','chi','exp']
	df_ = pd.DataFrame(np.array(vec),columns=nombres)
	df_.to_csv('datos_algoritmo.csv')
	df_.head()
	#df_.rename(columns={'col1': 'laSH', 'col2': 'mass', 'col3': 'cross_section', 'col4': 'chi2'}, inplace=True)
	df_= df_[df_['chi2'] <= critical_chi_sq]
	#df_filtrado.to_csv('datos_aleatorios.csv')

	return df_

if __name__ == '__main__': 
	#vec = aleatorio()
	df = algoritmo_G()
	#aleatorio()
	print("Finalizado")
	#Realiza el grafico respectivo.
	plt.figure()
	plt.plot(df['mass'],df['lash'],'.')
	plt.show()