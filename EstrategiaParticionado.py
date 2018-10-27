from abc import ABCMeta,abstractmethod
import numpy as np

class Particion():

  # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor 

  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  listaPartic = [] 
  numParticiones = 0
  porcentajeT = 0

  def __init__(self, porcentajeTrain, numeroParticiones):
    nombreEstrategia = "Validacion Simple"
    self.numParticiones = numeroParticiones
    self.porcentajeT = porcentajeTrain

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    for i in range(self.numParticiones):
      p=np.random.permutation(datos.numFilas)
      np.random.seed(seed)
      numFilas= datos.numFilas
      indice = int(self.porcentajeT*numFilas)
      #print(indice)
      aux = Particion()

      for i in range(indice):
        aux.indicesTrain.append(p[i])
      for i in range(indice, numFilas):
        aux.indicesTest.append(p[i])
      self.listaPartic.append(aux)

    return self.listaPartic
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  numParticiones = 0
  listaPartic = []

  def __init__(self, numeroParticiones)  :
    nombreEstrategia = "Validacion Cruzada"
    self.numParticiones = numeroParticiones
    for i in range(self.numParticiones):
      self.listaPartic.append(Particion())
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    np.random.seed(seed)
    nfilas = datos.numFilas
    p = np.random.permutation(nfilas)
    numDatosp = int(nfilas/self.numParticiones)
    auxPartic = []

    for i in range(self.numParticiones):
      auxPartic.append([])

      if self.numParticiones-1 == i:
        auxPartic[i] = p[i*numDatosp:]
      else:
        auxPartic[i] = p[i*numDatosp: i*numDatosp+numDatosp]

    for i in range(self.numParticiones):
      self.listaPartic[i].indicesTest = auxPartic[i].tolist()
      self.listaPartic[i].indicesTrain = []
      for j in range(self.numParticiones):
        if j != i:
          self.listaPartic[j].indicesTrain = auxPartic[j].tolist()
      
    pass
    

#####################################################################################################      
class ValidacionBootstrap(EstrategiaParticionado):

  def __init__(self, numeroParticiones)  :
    nombreEstrategia = "ValidacionCruzada"
    numParticiones = numeroParticiones
    listaPartic = []

  # Crea particiones segun el metodo de validacion por bootstrap.
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    posiciones=[]
    np.random.permutation(datos)

    for i in range(numParticiones):
      part = particion()
      aux = np.random.choice(datos.numFilas, datos.numfilas, replace=true)
      part.indicesTrain=aux
      aux=[]
      for i in range(datos.numFilas):
        if i not in aux:
          aux.append(i)
      part.indicesTest=aux
      listaPartic.add(part)
    return listaPartic

    
