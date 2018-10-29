# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from ClasificadorScity import ValidacionSimple
from ClasificadorScity import ValidacionCruzada
import numpy as np

errores=[]
print("Validacion Simple:")
errores.append(ValidacionSimple('german.data')[1])
errores.append(ValidacionSimple('german.data')[1])
errores.append(ValidacionSimple('german.data')[1])
print(errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviación Típica: ", d_tipica)
print()

errores=[]
print("Validacion Cruzada");
errores = ValidacionCruzada('german.data',0.1,3)[1]
print(errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviación Típica: ", d_tipica)