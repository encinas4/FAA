
from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np
from plotModel import plotModel 
import matpotlib.pyplot as plt 

data = 'ConjuntosDatos/wdbc.data'
dataset=Datos(data)
#estrategia= EstrategiaParticionado.ValidacionSimple(0.25,5)
estrategia= EstrategiaParticionado.ValidacionCruzada(10)
ii=estrategia.particiones[-1].indicesTrain 
#estrategia= EstrategiaParticionado.ValidacionBootstrap(10)
clasificador=Clasificador.ClasificadorVecinosProximos(51,False)
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace=1, seed=None)
plotModel(dataset.datos[ii,0],dataset.datos[ii,1],dataset.datos[ii,-1]!=0,clasificador,"Frontera",dataset.diccionarios) 
print("Probabilidad de error: ",errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviacion Tipica: ", d_tipica)