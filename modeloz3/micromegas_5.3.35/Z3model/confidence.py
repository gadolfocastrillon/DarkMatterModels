import numpy as np 
import pandas as pd 
import corner
import matplotlib.pyplot as plt 
from scipy.stats import chi2
import scipy.stats as stats

def best_fit_point_funtion(df):
	# Encuentra la fila con el valor mínimo de chi2
	best_fit_point = df[df['chi'] == df['chi'].min()]
	best_mass = best_fit_point['mass'].values[0]
	best_lash = best_fit_point['lash'].values[0]
	best_mu = best_fit_point['mu'].values[0]
	best_cross_section = best_fit_point['cross_section'].values[0]

	# Imprime el "best fit point"
	print("Best Fit Point:")
	print("Mass:", best_mass)
	print("Lash:", best_lash)
	print("Mu:", best_mu)
	print("Cross Section:", best_cross_section)
	print("Chi2:", best_fit_point['chi'].values[0])
	#print("Ratio:", best_fit_point['ratio'].values[0])
	return best_fit_point

def regions_confidence_2(df_, name="Confiabilidad_lash_mass", name1='mass', pass_1=0.01, name2='lash', pass_2=0.001):
    #Rangos en función del minimo y maximo de cada parametro. 
    a = df_[name1].min()
    b = df_[name1].max()
    c = df_[name2].min()
    d = df_[name2].max()

    #Cantidad de intervalos
    cantidad1 = abs((b-a))/pass_1
    cantidad2 = abs((d-c))/pass_2

    #Mensaje para mostrar en pantalla
    #print(r"El intervalo de " + name1 + " es: ("+str(a)+","+str(b)+") con un paso de: "+ str(pass_1)+ " total intervalos: "+str(cantidad1))
    #print(r"El intervalo de " + name2 + " es: ("+str(c)+","+str(d)+") con un paso de: "+ str(pass_2)+ " total intervalos: "+str(cantidad2))

    # Crear una cuadrícula de valores para name1 y name2
    name1_values = np.arange(a, b + pass_1, pass_1)
    name2_values = np.arange(c, d + pass_2, pass_2)

    #Elimino los valores de nan por cero. 
    df_[name1].fillna(0, inplace=True)
    df_[name2].fillna(0, inplace=True)

    # Utilizar pd.cut() para crear etiquetas basadas en las columnas 'x1' y 'x2'
    df_['group_1'] = pd.cut(df_[name1], bins=name1_values, labels=name1_values[:-1], include_lowest=True)
    df_['group_2'] = pd.cut(df_[name2], bins=name2_values, labels=name2_values[:-1], include_lowest=True)
    
   	#Elimino la columna "index" del dataframe. 
    df_ = df_.drop(columns=['index'])
    

    # Agrupa los datos del grupo1 y grupo2
    dat = df_.groupby(['group_1', 'group_2'])['ratio'].idxmax()
    best_fit_point = best_fit_point_funtion(df)
    
    # Encontrar el conjunto de datos con el mayor valor 'ratio' en cada grupo
    max_ratio_data = df_.loc[dat.dropna()]

    return max_ratio_data

def regions_confidence(df_, name="Confiabilidad_lash_mass", name1='mass', pass_1=0.01, name2='lash', pass_2=0.001):
    #Rangos en función del minimo y maximo de cada parametro. 
    a = df_[name1].min()
    b = df_[name1].max()
    c = df_[name2].min()
    d = df_[name2].max()

    #Cantidad de intervalos
    cantidad1 = abs((b-a))/pass_1
    cantidad2 = abs((d-c))/pass_2

    #Mensaje para mostrar en pantalla
    #print(r"El intervalo de " + name1 + " es: ("+str(a)+","+str(b)+") con un paso de: "+ str(pass_1)+ " total intervalos: "+str(cantidad1))
    #print(r"El intervalo de " + name2 + " es: ("+str(c)+","+str(d)+") con un paso de: "+ str(pass_2)+ " total intervalos: "+str(cantidad2))

    # Crear una cuadrícula de valores para name1 y name2
    name1_values = np.arange(a, b + pass_1, pass_1)
    name2_values = np.arange(c, d + pass_2, pass_2)

    #Elimino los valores de nan por cero. 
    df_[name1].fillna(0, inplace=True)
    df_[name2].fillna(0, inplace=True)

    # Utilizar pd.cut() para crear etiquetas basadas en las columnas 'x1' y 'x2'
    df_['group_1'] = pd.cut(df_[name1], bins=name1_values, labels=name1_values[:-1], include_lowest=True)
    df_['group_2'] = pd.cut(df_[name2], bins=name2_values, labels=name2_values[:-1], include_lowest=True)
    
   	#Elimino la columna "index" del dataframe. 
    df_ = df_.drop(columns=['index'])
    

    # Agrupa los datos del grupo1 y grupo2
    dat = df_.groupby(['group_1', 'group_2'])['ratio'].idxmax()
    best_fit_point = best_fit_point_funtion(df)
    
    # Encontrar el conjunto de datos con el mayor valor 'ratio' en cada grupo
    max_ratio_data = df_.loc[dat.dropna()]
    ########################################################################
    
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.scatter(max_ratio_data[name1], max_ratio_data[name2], c=max_ratio_data['ratio'], cmap='viridis')
    
    plt.scatter(best_fit_point[name1].values[0], best_fit_point[name2].values[0], c='red', marker='*', s=150, label='Max Exp')
    fig.colorbar(im,label='Ratio')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title(name)
    plt.ylabel(name2)
    plt.xlabel(name1)
    ax.set_facecolor('#440154FF')
    name_save = name + ".png"
    plt.savefig(name_save)
    plt.show()
    return max_ratio_data

def manipular_datos(df_):
	best_point = best_fit_point_funtion(df_)
	#print(best_point)
	chi_best = best_point['chi'].values[0]	
	df_['ratio'] = np.exp(-0.5*(df_['chi'] - chi_best))
	return df_

def graficador_cross_section(df):
	#Grafico sin interpolar.
	fig, ax = plt.subplots(figsize=(10,7)) 
	ax.set_xscale('log')
	ax.set_yscale('log')
	im = ax.scatter(df['mass'],df['cross_section'],c=df['mu'],cmap='viridis')
	fig.colorbar(im)
	plt.title("Singlete escalar con simetría $Z_{3}$")
	plt.ylabel("Cross Section ($\sigma_{SI}$)")
	plt.xlabel("Mass DM (GeV)")
	plt.savefig('mass_vs_cross_section.png')
	plt.show()

def contour_df(df_,name1='mass', pass_1=0.01, name2='lash', pass_2=0.001): 
	#Rangos en función del minimo y maximo de cada parametro. 
    a = df_[name1].min()
    b = df_[name1].max()
    c = df_[name2].min()
    d = df_[name2].max()

    #Cantidad de intervalos
    cantidad1 = abs((b-a))/pass_1
    cantidad2 = abs((d-c))/pass_2

    #Mensaje para mostrar en pantalla
    #print(r"El intervalo de " + name1 + " es: ("+str(a)+","+str(b)+") con un paso de: "+ str(pass_1)+ " total intervalos: "+str(cantidad1))
    #print(r"El intervalo de " + name2 + " es: ("+str(c)+","+str(d)+") con un paso de: "+ str(pass_2)+ " total intervalos: "+str(cantidad2))

    # Crear una cuadrícula de valores para name1 y name2
    name1_values = np.arange(a, b + pass_1, pass_1)
    name2_values = np.arange(c, d + pass_2, pass_2)

    X, Y = np.meshgrid(name1_values,name2_values)
    print("Meshgrid de X")
    print(X)
    print("El tamaño es:",len(X))


def f(X,Y,df):
	#Elimino los valores de nan por cero. 
    df_[name1].fillna(0, inplace=True)
    df_[name2].fillna(0, inplace=True)

    # Utilizar pd.cut() para crear etiquetas basadas en las columnas 'x1' y 'x2'
    df_['group_1'] = pd.cut(df_[name1], bins=name1_values, labels=name1_values[:-1], include_lowest=True)
    df_['group_2'] = pd.cut(df_[name2], bins=name2_values, labels=name2_values[:-1], include_lowest=True)
    
   	#Elimino la columna "index" del dataframe. 
    df_ = df_.drop(columns=['index'])
    

    # Agrupa los datos del grupo1 y grupo2
    dat = df_.groupby(['group_1', 'group_2'])['ratio'].idxmax()
    best_fit_point = best_fit_point_funtion(df)
    
    # Encontrar el conjunto de datos con el mayor valor 'ratio' en cada grupo
    max_ratio_data = df_.loc[dat.dropna()]


if __name__ == '__main__':
	nombres = ['index','lash','mass','mu','cross_section','chi']
	#Cargado de archivos
	df1 = pd.read_csv("archivo_profile_randtobest1exp.csv",names=nombres)
	df1 = df1.drop(columns=['index'])
	df2 = pd.read_csv("archivo_profile.csv",names=nombres)
	df2 = df2.drop(columns=['index'])
	#Concatenando los archivos y reseteando valores.
	df_concat = pd.concat([df1, df2], axis=0)
	df_concat = df_concat.reset_index() #Reseteamos los indices del dataframe. 
	df_concat = df_concat.drop_duplicates() #Eliminamos los datos duplicados.
	
	fig, axs = plt.subplots(1,2,figsize=(20,5))
	
	df_concat = df_concat[df_concat['mass'] > 1.5]
	
	df_concat = df_concat[df_concat['mu'] < 3.61]
	#axs[0].set_xscale('log')
	#axs[0].set_yscale('log')
	#im = axs[0].scatter(df_concat['mass'],df_concat['cross_section'],c=df_concat['mu'],cmap='viridis')
	#fig.colorbar(im)
	df_concat = df_concat[df_concat['mu']< 2*df_concat['mass']]
	#axs[1].set_xscale('log')
	#axs[1].set_yscale('log')
	#im = axs[1].scatter(df_concat['mass'],df_concat['cross_section'],c=df_concat['mu'],cmap='viridis')
	#fig.colorbar(im)
	#plt.show()

	df = manipular_datos(df_concat) #Crea los valores del ratio
	#contour_df(df)

	df_modif = regions_confidence_2(df)
	print("El tamaño de los datos es")
	print(len(df))
	print("El tamaño de los datos modificado")
	print(len(df_modif['ratio']))


	
	#print("El tamaño del dataframe es: ",len(df)) 
	#best_fit_point(df)
	#print(df.head())
	#graficador_cross_section(df)
	#regions_confidence(df, name="Frecuentista mass vs lash", name1='mass', pass_1=0.1, name2='lash', pass_2=0.01)
	#regions_confidence(df, name="Frecuentista mass vs mu", name1='mass', pass_1=0.01, name2='mu', pass_2=0.1)
	#regions_confidence(df, name="Frecuentista mass vs mu", name1='mass', pass_1=0.1, name2='cross_section', pass_2=1e-12)





'''
def graficador(df,fila_maxima,grupo1='mass',grupo2='lash',nombre="Analisis mass vs lash"):
	#Grafico sin interpolar.
	fig, ax = plt.subplots(figsize=(10,7)) 
	ax.set_xscale('log')
	ax.set_yscale('log')
	im = ax.scatter(df[grupo1],df[grupo2],c=df['ratio'],cmap='viridis')
	fig.colorbar(im)
	plt.scatter(fila_maxima[grupo1], fila_maxima[grupo2], c='red', marker='*', s=200, label='Max Exp')
	plt.title(nombre)
	plt.ylabel(grupo2)
	plt.xlabel(grupo1)
	plt.savefig('confidence_1.svg')
	plt.show()

def graficador_cross_section(df):
	#Grafico sin interpolar.
	fig, ax = plt.subplots(figsize=(10,7)) 
	ax.set_xscale('log')
	ax.set_yscale('log')
	im = ax.scatter(df['mass'],df['cross_section'],c=df['mu'],cmap='viridis')
	fig.colorbar(im)
	plt.title("Singlete escalar con simetría $Z_{3}$")
	plt.ylabel("Cross Section ($\sigma_{SI}$)")
	plt.xlabel("Mass DM (GeV)")
	plt.savefig('mass_vs_cross_section.svg')
	plt.show()
'''