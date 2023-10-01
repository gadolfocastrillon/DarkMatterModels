import numpy as np 
import subprocess 
from scipy.optimize import differential_evolution
from scipy.stats import chi2 
import matplotlib.pyplot as plt 
import time 

seed = 10 
def writer(file,dictionary):
	data1=open(file,'w')
	for items in dictionary.items(): 
		data1.write("%s %s\n"%items)
	data1.close()

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

	
	dato1 = subprocess.getoutput(COMMANDms) #ejecutar el comando desde la terminal 
	dato2 = subprocess.getoutput(COMMANDmp)
	#print(dato1,dato2)
	print(dato1)
	print(dato2)
	
	

	

if __name__ == '__main__':
	#print(seed)
	np.random.seed(seed)
	#print(seed)
	print("Running de_scan")
	
	X = [0] * (5)

	X[0] = np.random.uniform(0,4)
	X[1] = np.random.uniform(0,4)
	X[2] = np.random.uniform(-4,1)
	X[3] = np.random.uniform(0,1)
	X[4] = np.random.uniform(0,1)
	print(X)
	omega(X)