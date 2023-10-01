import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
nombres = ['lash','mass','mu','cross_section','chi']
df = pd.read_csv("archivo_profile.csv",names=nombres)
df = df.reset_index()
#Falta ejecutar el método para otro metodo diferentes y generar un espacio mas grande en los valores. 

#print(df.head())

plt.figure(figsize=(10,9))
plt.plot(df['mass'],df['lash'],'.')
plt.show()
