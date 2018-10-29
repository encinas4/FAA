from abc import ABCMeta,abstractmethod
import numpy as np
import scipy.stats as norm
import warnings

class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
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
      evaluacion = clasificador.clasifica(test,dataset.nombreAtributos, dataset.diccionarios)
      errores.append(self.error(test, evaluacion))

    return errores
    pass    
       
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):
  listaMatrices = []
  def _init_(self):
    self.listaMatrices=[]


  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario,laplace=0):
    #Creamos una matriz de frecuencias por cada atributo
    for x in range(len(atributosDiscretos)):
      if atributosDiscretos[x] == "Nominal":
        matrix = np.empty([len(diccionario[x].keys()),2], dtype=float)
        matrix*=0

        # Recorremos todos los datos y aumentamos la celda correspondiente al atributo y a true o false
        countClase = 0
        trues=0
        falses=0
        for f in datostrain[:,x]:
          filaM=int(f)
          c=int(datostrain[filaM,-1])
          matrix[filaM, c] +=1
          if  c == 1:
            trues +=1
          else:
            falses+=1

        # Aplicamos la regla de Laplace
        if laplace == 0:
          matrix = matrix +1   
          falses += len(diccionario[x].keys())
          trues += len(diccionario[x].keys())
        matrix[:,0] = matrix[:,0]/ falses
        matrix[:,1] = matrix[:,1]/ trues

        self.listaMatrices.append(matrix)
      else:
        matrix = np.empty([2,2], dtype=float)
        false = []
        true = []
        for f in range(len(datostrain[:,x])):
          filaM=int(f)
          
          if  datostrain[filaM, -1] == 0:
            false.append(datostrain[filaM, x])
          else:
            true.append(datostrain[filaM, x])
        # Calculamos la media y la varianza
        matrix[0,0]= np.mean(false)
        matrix[0,1]= np.mean(true)
        matrix[1,1]= np.var(true)
        matrix[1,0]= np.var(false)
        self.listaMatrices.append(matrix)
    pass
    
     
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    np.seterr(divide='ignore',invalid='ignore')
    falses=np.count_nonzero(datostest[:, -1]==0)
    trues=np.count_nonzero(datostest[:, -1]==1)
    total = trues+falses
    valores = []

    for i in range(len(datostest[:,0])):
      resT = 1
      resF = 1
      for j in range(len(datostest[i])):
        if atributosDiscretos[j] == "Nominal":
          aux = listaMatrices[j]
          resF *= aux[datostest[i,j],0]
          resT *= aux[datostest[i,j],1]
        else:
          aux = self.listaMatrices[j]
          u = aux[0,0]
          v =  aux[1,0]
          resF *= norm.norm.pdf(datostest[i,j], u,v)
          aux = self.listaMatrices[j]
          u = aux[0,1]
          v =  aux[1,1]
          resT *= norm.norm.pdf(datostest[i,j], u,v)
      resT *= trues/total
      resF *= falses/total
      if resT> resF:
        valores.append(1)
      else:
        valores.append(0)
    return valores 