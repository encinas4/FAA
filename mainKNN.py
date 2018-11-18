"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np

data = 'ConjuntosDatos/wdbc.data'
dataset=Datos(data)
#estrategia= EstrategiaParticionado.ValidacionSimple(0.25,5)
estrategia= EstrategiaParticionado.ValidacionCruzada(10)
#estrategia= EstrategiaParticionado.ValidacionBootstrap(10)s
clasificador=Clasificador.ClasificadorVecinosProximos(51,False)
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace=1, seed=None)
print("Probabilidad de error: ",errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviacion Tipica: ", d_tipica)