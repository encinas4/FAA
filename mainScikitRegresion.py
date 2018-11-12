# -*- coding: utf-8 -*-
"""
@author: profesores faa
"""

from ClasificadorScikit import RegresionLogistica
import EstrategiaParticionado
import numpy as np


data='ConjuntosDatos/example1.data'
#estrategia= EstrategiaParticionado.ValidacionSimple(0.25,5)
estrategia= EstrategiaParticionado.ValidacionCruzada(10)
#estrategia= EstrategiaParticionado.ValidacionBootstrap(10)
errores = RegresionLogistica(data, 10, estrategia)
print(errores)
media = np.mean(errores)
d_tipica = np.std(np.transpose(np.array(errores)), axis=0)
print("Promedio de error: ", media)
print("Desviacion Tipica: ", d_tipica)