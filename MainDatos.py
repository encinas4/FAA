# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from EstrategiaParticionado import ValidacionCruzada
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorNaiveBayes

#dataset=Datos('f:/temp/german.data')
dataset=Datos('balloons.data')
#print(dataset.datos)
estrategia= ValidacionSimple(0.60,2)
clasificador=ClasificadorNaiveBayes()
#  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):

errores=clasificador.validacion(estrategia,dataset,clasificador, seed=None)
print(errores)