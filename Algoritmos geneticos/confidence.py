import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import chi2
import scipy.stats as stats

def regions_confidence(df,name="Confiabilidad x1 con x2",name1='x1',lim_ini_1=-5,lim_fin_1=5,pass_1=0.1,name2='x2',lim_ini_2=-5,lim_fin_2=5,pass_2=0.1):
	x1_bins = np.arange(lim_ini_1,lim_fin_1,pass_1)
	x2_bins = np.arange(lim_ini_2,lim_fin_2,pass_2)
	
	
	# Usar pd.cut() para crear etiquetas basadas en las columnas 'mass' y 'lash'
	df['group_1'] = pd.cut(df[name1], bins=x1_bins, labels=x1_bins[:-1], include_lowest=True)
	df['group_2'] = pd.cut(df[name2], bins=x2_bins, labels=x2_bins[:-1], include_lowest=True)
	df['ratio'] = np.exp(-0.5*df['chi'])
	indices_maximos = df.groupby([name1, name2])['ratio'].idxmax()
	indices_maximos = indices_maximos.dropna()
	# Accede a las filas completas usando los índices máximos.
	fm = df.loc[indices_maximos]
	indice_max = df['ratio'].idxmax()

	# Accede a la fila completa usando el índice.
	fila_maxima = df.loc[indice_max]
	#fm = fm.reset_index().drop(columns=['index'])
	fig, ax = plt.subplots(figsize=(10,7)) 
	#ax.set_xscale('log')
	#ax.set_yscale('log')
	im = ax.scatter(fm[name1],fm[name2],c=fm['ratio'],cmap='viridis')
	fig.colorbar(im)
	#plt.set_label('Ratio')
	plt.scatter(fila_maxima[name1], fila_maxima[name2], c='red', marker='*', s=200, label='Max Exp')
	plt.title(name)
	plt.ylabel(name2)
	plt.xlabel(name1)
	name_save = name + ".png"
	plt.savefig(name_save)
	plt.show()

if __name__ == '__main__': 
	# Color style for output sample points
	de_pts = "#91bfdb" # Diver scan
	rn_pts = "#fc8d59" # Random scan
	gd_pts = "#ffffbf" # Grid scan

	column_names = ['x1', 'x2', 'x3','x4','chi']
	df = pd.read_csv("archivo_profile.csv",names=column_names)
	df = df.reset_index()
	df = df.drop(columns=['index'])
	print(df.head())


	grades = 2  # Grados de libertad
	sigma_1_confianza = 0.6827  # Nivel de confianza del 1 sigma 
	sigma_2_confianza = 0.9545  # Nivel de confianza del 2 sigma
	regions_confidence(df,name="Confiabilidad x1 con x2",name1='x1',name2='x2')
	regions_confidence(df,name="Confiabilidad x1 con x3",name1='x1',name2='x3')
	regions_confidence(df,name="Confiabilidad x1 con x4",name1='x1',name2='x4')
	regions_confidence(df,name="Confiabilidad x2 con x3",name1='x2',name2='x3')
	regions_confidence(df,name="Confiabilidad x2 con x4",name1='x2',name2='x4')
	regions_confidence(df,name="Confiabilidad x3 con x4",name1='x3',name2='x4')


	'''
	df = df.drop(columns=['index'])
	df['exp'] = np.exp(-0.5*df['chi'])

	#df2 = pd.read_csv("archivo_profile_2.csv", names=nombres)
	ini_lash = 10**-4
	fin_lash = 10**0


	indice_max = df['exp'].idxmax()

	# Accede a la fila completa usando el índice.
	fila_maxima = df.loc[indice_max]
	print("El valor maximo de los datos es:")
	print(fila_maxima)

	df['ratio'] = np.exp(-0.5*(df['chi'] - fila_maxima['chi']))



	# Calcula el valor crítico chi-cuadrado (inverso de la CDF)
	sigma_1 = stats.chi2.ppf(sigma_1_confianza, grades)
	sigma_2 = stats.chi2.ppf(sigma_2_confianza, grades)

	# Definir los límites y pasos para las etiquetas
	pass_mass = 30 
	pass_lash = 0.002
	mass_bins = list(range(1, 10000, pass_mass))
	lash_bins = [10**(-4) + pass_lash * i for i in range(1001)]

	# Usar pd.cut() para crear etiquetas basadas en las columnas 'mass' y 'lash'
	df['mass_group'] = pd.cut(df['mass'], bins=mass_bins, labels=mass_bins[:-1], include_lowest=True)
	df['lash_group'] = pd.cut(df['lash'], bins=lash_bins, labels=lash_bins[:-1], include_lowest=True)


	print("El tamaño de los datos es:",len(df))


	indices_maximos = df.groupby(['mass_group', 'lash_group'])['ratio'].idxmax()
	indices_maximos = indices_maximos.dropna()

	# Accede a las filas completas usando los índices máximos.
	fm = df.loc[indices_maximos]
	fm = fm.reset_index().drop(columns=['index'])
	print(fm)
	'''
	'''
	#Grafico sin interpolar.
	fig, ax = plt.subplots(figsize=(10,7)) 
	ax.set_xscale('log')
	ax.set_yscale('log')
	im = ax.scatter(fm['mass'],fm['lash'],c=fm['ratio'],cmap='viridis')
	fig.colorbar(im)
	plt.scatter(fila_maxima['mass'], fila_maxima['lash'], c='red', marker='*', s=200, label='Max Exp')
	plt.title("Singlete escalar con simetría $Z_{3}$")
	plt.ylabel("Cross Section ($\sigma_{SI}$)")
	plt.xlabel("Mass DM (GeV)")
	plt.savefig('mapeo_de_datos_likelihood-3.svg')
	plt.show()
	'''
	'''
	#Grafico con la interpolación
	from scipy.interpolate import griddata

	lash_interpolate = np.arange(fm['lash'].min(),fm['lash'].max(),pass_lash)
	mass_interpolate = np.linspace(fm['mass'].min(), fm['mass'].max(), num=len(lash_interpolate))
	points = np.column_stack((fm['mass'], fm['lash']))
	grid_z0 = griddata(points, fm['ratio'], (mass_interpolate, lash_interpolate), method='nearest')
	#grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
	#grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
	ig, ax = plt.subplots(figsize=(12, 8))

	plt.imshow(grid_z0.T, extent=(mass_interpolate.min(), mass_interpolate.max(), lash_interpolate.min(), lash_interpolate.max()), origin='lower', cmap='viridis', aspect='auto')
	plt.colorbar(label='Ratio')
	plt.title('Interpolated')

	plt.tight_layout()
	'''