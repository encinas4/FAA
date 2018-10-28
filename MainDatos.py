# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from EstrategiaParticionado import ValidacionCruzada
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionBootstrap
from Clasificador import ClasificadorNaiveBayes

#dataset=Datos('f:/temp/german.data')
dataset=Datos('balloons.data')
#print(dataset.datos)
estrategia= ValidacionSimple(0.25,3)
estrategia1= ValidacionCruzada(3)
estrategia2= ValidacionBootstrap(3)
clasificador=ClasificadorNaiveBayes()

errores=clasificador.validacion(estrategia,dataset,clasificador, seed=None)
print(errores)