import numpy as np 
import subprocess 
from scipy.optimize import differential_evolution
from scipy.stats import chi2 
import matplotlib.pyplot as plt 
import time 

dim = 4
sltns = 0 
min_chi_sq = 0.
alpha = 0.05
critical_chi_sq = chi2.isf(alpha,2)
seed = 10


lash_min = -4 #0.0001
lash_max = 0 #1 
yp_min = 0 
yp_max = 1 
ys_min = 0 
ys_max = 1
mass1_min = 1 #10 
mass1_max = 4 #10000
mass2_min = 1
mass2_max = 4


def writer(file,dictionary):
	data1=open(file,'w')
	for items in dictionary.items(): 
		data1.write("%s %s\n"%items)
	data1.close()

def samples_inside(x,chi_sq): 
    delta_chi_sq = chi_sq - min_chi_sq
    inside = delta_chi_sq <= critical_chi_sq
    return x[:, inside]

#X = [Ms2,Mp1,laS,ys,yp] orden de los parametros a ingresar en el vector
def omega(X):
	#---------------- Rutas de los archivos -------------------------------
	ruta = 'data.dat' #Ruta para guardar el archivo.
	rutaG = './main data.dat > temporal.dat' #Ruta para ejecutar micromegas 
	#----------------------------------------------------------------------
	
	#------Diccionario con los datos del archivo. -------------------------
	data = {'MH':125, 'Ms2':0.07,'Mp1':0.1,'laS':300,'ys':1000,'yp':173.2} 
	data['Ms2'] = 10**X[0]
	data['Mp1'] = 10**X[1]
	data['laS'] = 10**X[2]
	data['ys'] = 10**X[3]
	data['yp'] = 10**X[4]
	#----------------------------------------------------------------------

	#--------------------Sistema de escritura en el archivo----------------
	writer(ruta,data) 
	#----------------------------------------------------------------------

	#--------------------Obtener el valor de la densidad reliquia --------
	omeg = 0.0 
	#Corriendo micromegas y extrayendo la densidad reliquia 
	subprocess.getoutput(rutaG)
	#Comando para ejecutar una busqueda desde la terminal 
	#primero grep busca Omega en temporal.dat 
	#Despues awk separa la linea por espacios y me señala la tercer columna 
	COMMANDms = "grep 'Omega_1h^2' temporal.dat | awk 'BEGIN{FS=\"=\"};{print $2}'"  
	COMMANDmp = "grep 'Omega_2h^2' temporal.dat | awk 'BEGIN{FS=\"=\"};{print $2}'"

	def reliq_density(COMMAND):
		dato = subprocess.getoutput(COMMAND) #ejecutar el comando desde la terminal 
		if (len(dato)>0): 
			omeg = float(dato) 
		else: 
			omeg = -1 
		return omeg

	return [reliq_density(COMMANDms),reliq_density(COMMANDmp)]

	#----------------------------------------------------------------------

def gaussian(X):
	x = omega(X)
	#print("densidades reliquias",x[0],x[1])
	sigmams = np.sqrt((0.1*x[0])**2 + (0.001)**2) #Defino el sigma 
	sigmamp = np.sqrt((0.1*x[1])**2 + (0.001)**2) #Defino el sigma 
	xobs = 0.120 #densidad reliquia observable
	return (((x[0]-xobs)/sigmams)**2) + (((x[1]-xobs)/sigmamp)**2)


def de_scan(dim,round_to_nearest=None): 
	x = [] 
	chi_sq = []
	cs =[] 
	#[Ms2,Mp1,laS,ys,yp]
	bounds = [(mass1_min,mass1_max),(mass2_min,mass2_max),(lash_min,lash_max),(ys_min,ys_max),(yp_min,yp_max)] #ligaduras
	def objective(x_):
		chi_sq_ = gaussian(x_)
		chi_sq.append(chi_sq_)
		x_[0] = 10**x_[0]
		x_[1] = 10**x_[1]
		x_[2] = 10**x_[2]
		x_[3] = 10**x_[3]
		x_[4] = 10**x_[4]
		x.append(x_)
		#cross = csection()
		#print(cross)
		#cs.append(cross)
		#print(x_,cross)
		if (len(x) % 10 == 0):
			print(len(x))
		return chi_sq_

	#strategy condiciones a cumplir la estrategia 
	#popsize cantidad de individuos cantidatos a generar 
	#tol tolerancia del problema? error del problema? 
	#mutation capacidad de mutación de los datos al partir de 2 padres y crear un hijo 
	#La mutación esta entre un minimo y un maximo de mutación (min,max)
	#recombination es conocida como probabilidad de cruce. 
	#El aumento de este valor permite que una mayor cantidad de mutates progresen a la siguiente generación
	#pero a riegos de la estabilidad de la población.
	#Especificar seed para minimizaciones repetibles.
	#Polish se usa para pulir el mejor miembro de la población final, mejora ligeramente la minimización.
	#Para problemas grandes con muchas restricciones, el pulido puede llevar mucho tiempo debido a los calculos jacobianos.
	
	differential_evolution(objective, bounds,
                           strategy='currenttobest1bin', maxiter=None,
                           popsize=50, tol=0.01, mutation=(1.5, 1.99999), recombination=0.9,
                           polish=False, seed=seed)

	#rand1bin, 50, 0.01, 1.5-1.99, 0.9 seed = 10 guardado en 5 
	#currenttobest1exp, 50, 0.01, 1.5-1.99, 0.9 seed = 16 guardado en 12 

	valor = np.array(x).T
	datos=[valor[i] for i in range(len(valor))]
	datos.append(cs) #Agrego cross section a 

	if round_to_nearest is not None: 
		len_x = len(x)
		keep_n = len_x - (len_x %round_to_nearest)
		x = x[:keep_n]
		chi_sq = chi_sq[:keep_n] 
	
	return samples_inside(np.array(datos),np.array(chi_sq)),len(x)
	#return samples_inside(np.array(x).T, np.array(chi_sq)),len(x)
	#return x
#---------------------------------------------------------------------------
'''
Pendiente: 
Falta guardar los datos, pasar los valores de picmetro² a cm² con 1picometro = 1e-10 cm 
pico² = 1e-20 cm² 
verificar que se guardan los datos del cross section. 

'''
#------------------------------Uso en la función main-----------------------
if __name__ == '__main__':
	#print(seed)
	np.random.seed(seed)
	#print(seed)
	print("Running de_scan")
	
	 
	t0 = time.time()
	x, calls  = de_scan(dim) 	
	#calls me devuelve el tamaño de los datos 
	#x es un arreglo con: x[0]:lambdahs, x[1]: MasaS, x[2]: Mu3, x[3]: cross section.
	writer2(x.T,"datosLikelihood-14.txt") 
	#datosLikelihood-12 de interes
	#datosLikelihood-8 de interes
	#datosLikelihood-9 de interes
	de_time = time.time() - t0 
	print(r'Tiempo de ejecución : '+ str(de_time)) #Imprime el tiempo que demora en ejecutar el progama.
	print("Finalizado")
