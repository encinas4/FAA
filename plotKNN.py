
from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np
from plotModel import plotModel
import matplotlib.pyplot as plt

data = 'ConjuntosDatos/example4.data'
dataset=Datos(data)
#estrategia= EstrategiaParticionado.ValidacionSimple(0.25,5)
estrategia= EstrategiaParticionado.ValidacionCruzada(10)
#estrategia= EstrategiaParticionado.ValidacionBootstrap(10)

clasificador=Clasificador.ClasificadorVecinosProximos(3,True)
#clasificador=Clasificador.ClasificadorRegresionLogistica(100)

errores=clasificador.validacion(estrategia,dataset,clasificador,laplace=1, seed=None)
print("Probabilidad de error: ",errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviacion Tipica: ", d_tipica)

#plot
ii=estrategia.listaPartic[-1].indicesTrain
plotModel(dataset.datos[ii,0],dataset.datos[ii,1],dataset.datos[ii,-1]!=0,clasificador,"Frontera",dataset.nominalAtributos,dataset.diccionarios, dataset.extraeDatos(estrategia.listaPartic[-1].indicesTrain))
plt.figure()
plt.plot(dataset.datos[dataset.datos[:,-1] == 0,0], dataset.datos[dataset.datos[:,-1] == 0,1], 'bo')
plt.plot(dataset.datos[dataset.datos[:,-1] == 1,0], dataset.datos[dataset.datos[:,-1] == 1,1], 'ro')
plt.show()