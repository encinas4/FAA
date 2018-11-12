# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from ClasificadorScikit import VecinosProximos
import EstrategiaParticionado
import numpy as np


data='ConjuntosDatos/wdbc.data'
#estrategia= EstrategiaParticionado.ValidacionSimple(0.25,5)
#estrategia= EstrategiaParticionado.ValidacionCruzada(10)
estrategia= EstrategiaParticionado.ValidacionBootstrap(10)
errores = VecinosProximos(data, 10, estrategia)
print(errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviacion Tipica: ", d_tipica)