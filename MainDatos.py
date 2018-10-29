# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np

dataset=Datos('german.data')
estrategia= EstrategiaParticionado.ValidacionCruzada(4)

clasificador=Clasificador.ClasificadorNaiveBayes()
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace=1, seed=None)
print("Probabilidad de error: ",errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviación Típica: ", d_tipica)