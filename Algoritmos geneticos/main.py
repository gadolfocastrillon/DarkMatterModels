import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import algoG as al
import confidence as con

def cargado(nombre):
	column_names = ['x1', 'x2', 'x3','x4','chi']
	df = pd.read_csv(nombre,names=column_names)
	df = df.reset_index()
	df = df.drop(columns=['index'])
	return df 

if __name__ == '__main__':
	dim = 4  
	#x,calls_x = al.de_scan(dim,estrategia= 'rand1bin',nombre_archivo='archivo_profile_rand1bin.csv')
	#y,calls_y = al.de_scan(dim,estrategia= 'currenttobest1exp',nombre_archivo='archivo_profile_currenttobest1exp.csv')
	#z,calls_z = al.de_scan(dim,estrategia= 'best2exp',nombre_archivo='archivo_profile_best2exp.csv')

	df_x = cargado('archivo_profile_rand1bin.csv')
	df_y = cargado('archivo_profile_currenttobest1exp.csv')
	df_z = cargado('archivo_profile_best2exp.csv')

	df = pd.concat([df_x, df_y, df_z], axis=0)

	#print(df.head())
	con.regions_confidence(df)
	#regions_confidence(df_y)
	#regions_confidence(df_z)
	#al.graficador(x[0],x[1])
	#al.graficador(y[0],y[1])
	#al.graficador(z[0],z[1])
