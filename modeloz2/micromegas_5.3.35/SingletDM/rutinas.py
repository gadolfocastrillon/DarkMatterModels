import numpy as np 
import subprocess
import matplotlib.pyplot as plt 
import pandas as pd 

RUTA = 'data.dat'
RUTA_EJECUTABLE = './main data.dat > temporal.dat'
data = {'Q': 14, 'Mh': 120, 'laS': 0.1, 'laSH': 0.15, 'Mdm1': 700}

def writer(file,dictionary): 
	data = open(file,'w')
	for items in dictionary.items(): 
		data.write("%s %s\n"%items)
	data.close() 

def mod_archivo(X):
	data['Mdm1'] = X[1] #Valores de la masa de la particulas.
	data['laSH'] = X[0] #Valores de laSH
	writer(RUTA,data)

def funcion_float(val):
	if(len(val)>0):
		return float(val)
	else: 
		return -1 

def calc_densidad_reliquia(): 
	omeg = 0.0 
	subprocess.getoutput(RUTA_EJECUTABLE)
	COMMAND = "grep 'Omega' temporal.dat | awk 'BEGIN{FS=\"=\"};{print $3}'"
	val = subprocess.getoutput(COMMAND)
	return funcion_float(val)

def calc_cross_section():
	cross = 0.0 
	COMMAND = "grep 'proton  SI' temporal.dat | awk '{print $3}'"
	val = subprocess.getoutput(COMMAND)
	return funcion_float(val)

def omega(theta): 
	mod_archivo(theta)
	omeg = calc_densidad_reliquia()
	return omeg


def gaussian(theta): 
	x = omega(theta) 
	sigma = np.sqrt((0.1*x)**2 + (0.001)**2) #Defino el sigma 
	xobs = 0.120 #densidad reliquia observable
	val = ((x-xobs)/sigma)**2
	return  val,np.exp(-(1/2)*val) #Gaussiana

def printf_(x): 
	if (len(x)%1000 == 0): 
		print(len(x),end='\r')
