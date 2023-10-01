import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import chi2
import scipy.stats as stats

nombres = ['lash','mass','cross_section','rq']

df1 = pd.read_csv("datos_aleatorios.csv",names=nombres)
df2 = pd.read_csv("datos_aleatorios_2.csv", names=nombres)

ini_lash = 10**-4
fin_lash = 10**1
mass_ini = 0
mass_fin = 2.30

def gaussiana(x): 
	sigma = np.sqrt((0.1*x)**2 + (0.001)**2) #Defino el sigma 
	xobs = 0.120 #densidad reliquia observable
	val = ((x-xobs)/sigma)**2 #Gaussiana
	return val, np.exp(-(1/2)*val)

gl = 1  # Grados de libertad
sigma_1_confianza = 0.6827  # Nivel de confianza del 1 sigma 
sigma_2_confianza = 0.9545  # Nivel de confianza del 2 sigma

df = pd.concat([df1, df2], axis=0)

# Calcula el valor crítico chi-cuadrado (inverso de la CDF)
sigma_1 = stats.chi2.ppf(sigma_1_confianza, gl)
sigma_2 = stats.chi2.ppf(sigma_2_confianza, gl)

datos_chi = []
datos_exp = []
for val in df['rq']:
	val1, val2 = gaussiana(val)
	datos_chi.append(val1)
	datos_exp.append(val2)
	#print(val2)

df['chi'] = datos_chi
df['exp'] = datos_exp

intervalos_masa = pd.interval_range(start=1, end=2700,freq=1)
intervalos_lash = pd.interval_range(start=ini_lash,end=fin_lash,freq=ini_lash)


df['zona_masa'] = pd.cut(df['mass'],bins=intervalos_masa)
df['zona_lash'] = pd.cut(df['lash'],bins=intervalos_lash)


x_mass = range(1,2700,1)
x_lash = [] 


grupos1 = df.groupby('zona_masa')
grupos2 = df.groupby('zona_lash')


max_exp_mass = []
max_exp_lash = [] 



for grupo,datos in grupos1: 
	max_exp_ = datos['exp'].max()
	max_exp_mass.append(max_exp_)

x_ = ini_lash

for grupo,datos in grupos2:
	max_exp_ = datos['exp'].max()
	max_exp_lash.append(max_exp_)
	x_lash.append(x_)
	x_+= ini_lash



fig = plt.figure(figsize=(12, 8))
plt.title("Profile likelihood para modelo z2")
# Gráfico 1 (arriba a la izquierda)
ax1 = plt.subplot(2, 2, 1)  # 2 filas, 2 columnas, primer gráfico
ax1.plot(df['mass'], df['lash'], 'k.')
ax1.set_title('Graphics study of the parameters')
ax1.grid()
ax1.set_xlabel('Mass (Gev)')
ax1.set_ylabel('laSH')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(20, 1000)

# Gráfico 2 (arriba a la derecha)
ax2 = plt.subplot(2, 2, 2)  # 2 filas, 2 columnas, segundo gráfico
ax2.plot(x_lash, max_exp_lash, 'b.')
ax2.axhline(y=sigma_1_confianza, color='red', linestyle='--', label='y=15')
ax2.axhline(y=sigma_2_confianza, color='green', linestyle='--', label='y=15')
ax2.set_yscale('linear')
ax2.set_title('laSH interval 1D')
ax2.set_xlabel('laSH')
ax2.set_ylabel('Likelihood')
ax2.set_xscale('log')

# Gráfico 3 (abajo a la izquierda)
ax3 = plt.subplot(2, 2, 3)  # 2 filas, 2 columnas, tercer gráfico
ax3.plot(x_mass, max_exp_mass, 'b.')
ax3.axhline(y=sigma_1_confianza, color='red', linestyle='--', label='y=15')
ax3.axhline(y=sigma_2_confianza, color='green', linestyle='--', label='y=15')
ax3.set_title('Mass interval 1D')
ax3.set_xlabel('mass')
ax3.set_ylabel('Likelihood')
ax3.set_xscale('log')

# Gráfico 4 (abajo a la derecha)
ax4 = plt.subplot(2, 2, 4)  # 2 filas, 2 columnas, cuarto gráfico
ax4.set_facecolor('black')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(40,80)
im = ax4.scatter(df['mass'], df['lash'], c=df['exp'], cmap='viridis')
barra_colores = fig.colorbar(im)
barra_colores.set_label('Likelihood')
ax4.set_xlabel('Mass (Gev)')
ax4.set_ylabel('laSH')
ax4.set_title("Profile likelihood")
plt.subplots_adjust(hspace=0.5)
plt.savefig("grafico con limites.png")

plt.show()