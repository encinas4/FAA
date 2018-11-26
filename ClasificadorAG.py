from abc import ABCMeta,abstractmethod
import numpy as np
import scipy.stats as norm
import warnings
import math as math
import collections as collections 
import copy

class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  listaMatrices = []
  def _init_(self):
    self.listaMatrices=[]
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error  
    return np.count_nonzero(datos[:, -1] != pred)/len(pred)
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,laplace=0,seed=None):
    errores = []
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
    #Creamos las particiones
    particionado.creaParticiones(dataset, seed)
    # Recorremos las particiones
    for particion in particionado.listaPartic:
      #Extraemos los datos de las particiones, tanto de test como de train
      train = dataset.extraeDatos(particion.indicesTrain)
      test = dataset.extraeDatos(particion.indicesTest)

      # Entrenamos con los datos de train y evaluamos con los datos de test
      entrenamiento = clasificador.entrenamiento(train, dataset.tipoAtributos, dataset.diccionarios,laplace)
      evaluacion = clasificador.clasifica(test,dataset.tipoAtributos, dataset.diccionarios)
      errores.append(self.error(test, evaluacion))
      clasificador.calcularMatrizConfusion(test, evaluacion)

    return errores


  def calcularMedDesv(self, train):
    for i in range(len(train[0]-1)):
      sum = np.sum(train[:,i])
      aux=[]
      aux.append(sum/len(train[:,i]))
      aux.append(np.std(train[:,i]))
      self.listaMatrices.append(aux)
    pass


  def normalizarDatos(self,datos, tipoAtributos):
    for i in range(len(datos[0])-1):
      if tipoAtributos[i] == "Continuo":
        datos[:,i]= (datos[:,i] - self.listaMatrices[i][0])/self.listaMatrices[i][1]
    pass

  def validacion(self,particionado,dataset,clasificador, laplace=0 , seed=None, constAprend = 1, nepocas = 1):
    aux = []
    particionado.creaParticiones(dataset, seed)

    for i in range(particionado.numParticiones):
      train = dataset.extraeDatos(particionado.listaPartic[i].indicesTrain)
      test = dataset.extraeDatos(particionado.listaPartic[i].indicesTest)
      if norm:
        clasificador.calcularMedDesv(test)
        clasificador.normalizarDatos(test, dataset.tipoAtributos)

      clasificador.entrenamiento(train)
      if norm:
        clasificador.normalizarDatos(train, dataset.tipoAtributos)

      clases = clasificador.clasifica(test,train, dataset.nominalAtributos,dataset.diccionarios)
      aux.append(clasificador.error(test, clases))
    return aux




class ClasificadorAlgoritmoGenetico(Clasificador):
  listaMatrices=[]#0 media 1 std
  
  datosTrain = None
  poblacion = 0
  epocas = 0
  fitnes = 0.0

  #constructor de vecinos, k=vecinos, norm=true o false si se quiere normalizar
  def __init__(self, poblacion, epocas, fitnes):
    self.epocas = epocas
    self.poblacion = poblacion
    self.fitnes = fitnes

  def void entrenamiento(self, dataset):
    datos = np.empty([len(datos[0]),len(datos)], dtype=int)
    datos[:,-1] = dataset[:,-1]
    for i in range(len(datos[0])-1):
      int K = 1 + 3.322*np.log10(len(dataset))
      int a = (np.amax(datos[:,i]) - np.amin(datos[:,i]))*K
      for j in range(len(datos)):
        datos[j,i] = (int(dataset[j,i]/a)+1)
    dataset = datos
    pass

