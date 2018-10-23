from abc import ABCMeta,abstractmethod


class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones  
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
  

  def __init__(self, porcentajeTrain, numeroParticiones):
    nombreEstrategia = "ValidacionSimple"
    numParticiones = numeroParticiones
    porcentajeT = porcentajeTrain
    listaPartic = [] 



  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):

    for i in range(numParticiones):
      np.random.permutation(datos)
      indice = int(porcentajeT*len(datos))
      aux = Particion()
      aux.indicesTrain.concat(datos[:indice])
      aux.indicesTest.concat(datos[indice+1:])
      listaPartic.add(aux)
      print(aux)

    return listaPartic
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):

  def __init__(self, numeroParticiones)  :
    nombreEstrategia = "ValidacionCruzada"
    numParticiones = numeroParticiones
    listaPartic = []

  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    superior = int(len(datos)/numParticiones)
    base=superior
    inferior=0
    np.random.permutation(datos)
    for i in range(numParticiones):
      aux = Particion()
      aux.indicesTrain.concat(datos[inferior:superior-1])
      aux.indicesTest.concat(datos[superior:])
      if inferior != 0:
        aux.indicesTest.concat(datos[0:inferior-1])

      listaPartic.add(aux)
      print(aux)
      inferior=superior
      superior= superior+base
      
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
      aux = Particion()
        r1 = random.random(len(datos)/2)
        r2 = random.random(len(datos)/2)
      for i in range(len(datos)/2):
        x = int(r1[i]*len(datos))
        y = int(r2[i]*len(datos))
        aux.indicesTrain.concat(datos[x])
        aux.indicesTest.concat(datos[y])
      listaPartic.add(aux)
  return listaPartic

    
