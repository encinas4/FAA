from abc import ABCMeta,abstractmethod
import numpy as np
import scipy.stats as norm
import warnings
import math as math
import collections as collections 
import copy

class ClasificadorGenetico(object):

  # Clase abstracta
  __metaclass__ = ABCMeta
  listaMatrices = []
  def _init_(self):
    return

  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error  
    return np.count_nonzero(datos[:, -1] != pred)/len(pred)

  def validacion(self,ind, dataset, b, k):
    for dato in dataset.datos:
      clase=[]
      for regla in ind:  
        for i in range(len(dato)-1):
          if(not b):#no binario
            """if(regla[i] != int(dato[i]/k)):
              break
            if(regla[i] == int(dato[i]/k) and regla[-1] == dato[-1] and i == len(dato)-2): 
              acierto+=1
            if (i == len(dato)-2):
              error+=1"""
            aux =[]
            aux.append(dato[0:-2]/k+1)#para cada dato[i] lo dividimos por k y le sumamos 1
            if(cmp (regla[0:-2], aux)==1):
              clase.append(regla[-1])
          else:
            a = 1 
            #cosas
        if(clase != []):
          #cnt = Counter(list_of_integers)
          #if(cnt.most_common(1) == dato[-1])
          if (a == 1):
            acierto+=1
        total+=1
    return acierto/total





  

