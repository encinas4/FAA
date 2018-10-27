from abc import ABCMeta,abstractmethod
import numpy as np
import scipy.stats as norm

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
  def validacion(self,particionado,dataset,clasificador,seed=None):
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
      print(particion.indicesTrain)
      print(particion.indicesTest)
      #Extraemos los datos de las particiones, tanto de test como de train
     # train = dataset.extraeDatos(particion.indicesTrain)
    #  test = dataset.extraeDatos(particion.indicesTest)

      # Entrenamos con los datos de train y evaluamos con los datos de test
    #  entrenamiento = clasificador.entrenamiento(train, dataset.nombreAtributos, dataset.diccionarios)
    #  evaluacion = clasificador.clasifica(test,dataset.nombreAtributos, dataset.diccionarios)
     # errores.append(self.error(dataset.datos, evaluacion))

    #return errores
    pass    
       
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):
  listaMatrices = []
  def _init_(self):
    self.listaMatrices=[]


  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    #Creamos una matriz de frecuencias por cada atributo
    for x in range(len(atributosDiscretos)):
      if atributosDiscretos[x] == "Nominal":
        matrix = np.empty([len(diccionario[x].keys()),2], dtype=float)

        # Recorremos todos los datos y aumentamos la celda correspondiente al atributo y a true o false
        countClase = 0
        for f in datostrain[:,x]:
          filaM=f
          matrix[filaM, datostrain[countClase,-1]] +=1
          countClase += 1
          if  datostrain[countClase,-1] == 1:
            trues +=1
          else:
            flases+=1

        # Aplicamos la regla de Laplace
        if 0 in matrix:
          matrix = matrix +1   
          
        matrix[:,0] = matrix[:,0]/ falses
        matrix[:,1] = matrix[:,1]/ trues    

        self.listaMatrices.append(matrix)  
      else:
        matrix = np.empty([2,2], dtype=float)
        false = []
        true = []
        for f in datostrain[:,x]:
          filaM=int(f)
          
          if  datostrain[filaM, -1] == 0:
            false.append(datostrain[filaM, x])
          else:
            true.append(datostrain[filaM, x])
        matrix[0,0]= np.mean(false)
        matrix[0,1]= np.mean(true)
        matrix[1,1]= np.var(true)
        matrix[1,0]= np.var(false)
        self.listaMatrices.append(matrix)
        #primero media luego varianza
    pass
    
     
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    falses=np.count_nonzero(datostest[:, -1]==0)
    trues=np.count_nonzero(datostest[:, -1]==1)
    total = trues+falses
    valores = []

    for i in range(len(datostest[0,:])):
      resT = 1
      resF = 1
      for j in range(len(datostest[i])):
        if atributosDiscretos[j] == "Nominal":
          aux = listaMatrices[i]
          resF *= aux[datostest[i,j],0]
          resT *= aux[datostest[i,j],1]
        else:
          aux = self.listaMatrices[i]
          u = aux[0,0]
          v =  aux[1,0]
          resF *= norm.norm.pdf(datostest[i,j], u,v)
          aux = self.listaMatrices[i]
          u = aux[0,1]
          v =  aux[1,1]
          resT *= norm.norm.pdf(datostest[i,j], u,v)
      resT *= trues/total
      resF *= falses/total
      if resT> resF:
        valores.append(1)
      else:
        valores.append(0)
          #pos = valores.index(max(resT,resF))
    return valores 