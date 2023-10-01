import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear un DataFrame de ejemplo
data = {
    'mass': np.random.normal(10, 2, 100),
    'lash': np.random.normal(5, 1, 100),
    'ratio': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)

# Valor de 'ratio' para el cual deseas dibujar los contornos
valor_objetivo = 'B'

# Filtrar el DataFrame por 'ratio' deseado
df_filtered = df[df['ratio'] => valor_objetivo]

# Visualizar los puntos
plt.scatter(df['mass'], df['lash'], c=df['ratio'].map({'A': 'red', 'B': 'blue', 'C': 'green'}), label='Puntos', cmap='viridis')

# Dibujar contornos alrededor de los puntos con 'ratio' deseado
plt.scatter(df_filtered['mass'], df_filtered['lash'], c='none', edgecolors='black', marker='o', s=100, label=f'Ratio = {valor_objetivo}')
plt.xlabel('mass')
plt.ylabel('lash')
plt.legend()
plt.grid(True)
plt.title('Contornos para Ratio B')
plt.show()
