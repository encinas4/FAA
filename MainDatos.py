# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np

data = 'ConjuntosDatos/example1.data'
dataset=Datos(data)
#estrategia= EstrategiaParticionado.ValidacionSimple(0.25, 1)
estrategia= EstrategiaParticionado.ValidacionSimple(0.25,1)
#clasificador=Clasificador.ClasificadorVecinosProximos(1,True)
clasificador=Clasificador.ClasificadorRegresionLogistica(10)
errores=clasificador.validacion(estrategia,dataset,clasificador)
print("Probabilidad de error: ",errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviacion Tipica: ", d_tipica)