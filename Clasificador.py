from abc import ABCMeta,abstractmethod


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
    return np.count_nonzero(datos[:, -1] != pred)/pred.size
	
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
    particionado.creaParticiones(self, dataset, seed)
    error = np.array(())
    if(particionado.numParticiones == 1):
      entrenamiento = clasificador.entrenamiento(self, particionado[0].indicesTrain, dataset.NombreAtributos, dataset.diccionario)
      evaluacion = clasificador.clasifica(self, particionado[0].indicesTest,dataset.NombreAtributos, dataset.diccionario)
      return err = error(self, dataset, evaluacion)
    else:
      for particion in particionado.listaPartic:
        entrenamiento = clasificador.entrenamiento(self, particionado.indicesTrain, dataset.NombreAtributos, dataset.diccionario)
        evaluacion = clasificador.clasifica(self, particionado.indicesTest,dataset.NombreAtributos, dataset.diccionario)

	pass
       
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

 

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):

    #Creamos una matriz de frecuencias por cada atributo
    for x in len(atributosDiscretos):
      if atributosDiscretos[x] == "Nominal"
        matrix = np.empty([len(diccionario[x].keys()),2], dtype=float)

        countClase = 0
        for filaM in datostrain[:,x]:
          matrix[filaM, datostrain[countClase:-1]] +=1
          countClase += 1;
        
      else
        matrix = np.empty([2,2], dtype=float)







 #    matrixList.append(matrix)


	pass
    
     
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    pass

    
    





  