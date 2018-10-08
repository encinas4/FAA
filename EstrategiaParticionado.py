from abc import ABCMeta,abstractmethod


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
      aux.indicesTrain.concat(datos[:index])
      aux.indicesTest.concat(datos[index+1:])
      listaPartic.add(aux)
      print(aux)

    return listaPartic
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    pass
    

#####################################################################################################      
class ValidacionBootstrap(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion por bootstrap.
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    pass

    
