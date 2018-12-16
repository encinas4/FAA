from abc import ABCMeta,abstractmethod
import numpy as np
import scipy.stats as norm
import warnings
import math as math
import collections as collections 
import copy

class Intervalo():
  id = 0
  max = 0
  min = 0

  def __init__(self, id, min, max):
    self.id = id
    self.min = min
    self.max = max

class ClasificadorGenetico():
  listaMatrices = []

  def __init__(self):
    return

  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error  
    return np.count_nonzero(datos[:, -1] != pred)/len(pred)

  #Metodo que crea la tabla de intervalos a partir de un dataset y un k intervalos dado 
  def crearTablaIntervalos(self, dataset, train, k):
    column = []
    self.listaMatrices = []
    for a in range(len(dataset.nombreAtributos)-1): # recorremos los a atributos menos la clase
      column = train[:,a]   # extraemos toda la columna del atributo con el que trabajamos
      auxA = []             # lista auxiliar de los intervalos por atributo
      minim = min(column)
      maxim = max(column)
      A = (maxim -minim)/k
      
      intervalo = Intervalo(0, 0, 0)    # En el caso de intervalos 0
      auxA.append(intervalo)
      for i in range(k):    # #bucle que crea todos los intervalos de un atributo
        intervalo = Intervalo(i+1, minim, minim+A)
        auxA.append(intervalo)
        minim += A

      self.listaMatrices.append(auxA)    # introducimos los intervalos de un atributo en la lista de matrices


  def validacion(self,ind, dataset, train, b, k):
    # Creamos la lista de intervalos por atributo
    self.crearTablaIntervalos(dataset, train, k)
    total=0
    aciertos=0
    if(not b):
      for dato in train:
        clase=[]
        for regla in ind:  
          transDato=[]
          for i in range(len(dato)-1):
            transDato.append(self.idIntervalor(i, dato[i]))
          if (self.compararLista(regla[0:-1], transDato)):
            clase.append(regla[-1])  
        #print(clase)
        if clase !=[] and np.bincount(clase).argmax() == dato[-1]:
          aciertos+=1
         
        total+=1
    else:
      for dato in train:
        clase=[]
        for regla in ind:  
          transDato=[]
          for i in range(len(dato)-1):
              if (self.compararLista(regla[0:-1], transDato)):
                break
            clase.append(regla[-1])  
        #print(clase)
        if clase !=[] and np.bincount(clase).argmax() == dato[-1]:
          aciertos+=1
         
        total+=1

    return 1- aciertos/total
          
           
    
  def compararLista(self, a,b):
    if len(a) == len(b):
      for i in range(len(a)):
        if a[i] != b[i]:
          return False
    return True

  def idIntervalor(self, id, valor):
    for intervalo in self.listaMatrices[id]:
      if(intervalo.min <= valor and intervalo.max>= valor):
        return intervalo.id
    return intervalo.id

  def clasificar(self, individuo, dataset, train, b,k):
    self.crearTablaIntervalos(dataset, train, k)
    if(not b):
      pred=[]
      for dato in train:
        clase=[]
        for regla in individuo:  
          transDato=[]
          for i in range(len(dato)-1):
            transDato.append(self.idIntervalor(i, dato[i]))
          if (self.compararLista(regla[0:-1], transDato)):
            clase.append(regla[-1])
        if clase== []:  
          for r in individuo:
            clase.append(r[-1])
        
        pred.append(np.bincount(clase).argmax())
      return pred

  def error(self, pred, datos):
    return np.count_nonzero(datos[:, -1] != pred)/len(pred)