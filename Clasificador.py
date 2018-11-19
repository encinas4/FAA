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


##############################################################################
# CLASIFICADOR NAIVE BAYES

class ClasificadorNaiveBayes(Clasificador):
  listaMatrices = []
  listaMatricConfusion=[]
  def _init_(self):
    self.listaMatrices=[]
    self.listaMatricConfusion=[]

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
        if laplace == 1 and 0 in matrix:
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

  def calcularMatrizConfusion(self, datos, pred):
    aux = np.empty([2,2], dtype=int)
    aux*=0

    
    for i in range(len(datos[:,-1])):
      if pred[i]==1:
        if datos[i,-1]==1:
          aux[0,0]+=1
        else:
          aux[0,1]+=1
      else:
        if datos[i,-1]==1:
          aux[1,0]+=1
        else: 
          aux[1,1]+=1
    self.listaMatricConfusion.append(aux)
    pass

##############################################################################################
# CLASIFICADOR VECINOS PROXIMOS

class ClasificadorVecinosProximos(Clasificador):
  listaMatrices=[]#0 media 1 std
  norm=norm
  datosTrain = None
  k=0

  #constructor de vecinos, k=vecinos, norm=true o false si se quiere normalizar
  def __init__(self, k, norm):
    self.k=k
    self.norm=norm
    self.listaMatrices=[]#0 media 1 std
    self.datosTrain = None

  def validacion(self,particionado,dataset,clasificador, laplace=0 , seed=None, constAprend = 1, nepocas = 1):
    aux = []
    particionado.creaParticiones(dataset, seed)

    #se intera el numero de particiones que se ha anyadido
    for i in range(particionado.numParticiones):
      train = dataset.extraeDatos(particionado.listaPartic[i].indicesTrain)
      test = dataset.extraeDatos(particionado.listaPartic[i].indicesTest)

      clasificador.entrenamiento(train)
      if norm:
        copia = copy.copy(dataset)
        copia.datos = train
        clasificador.normalizarDatos(copia.datos, dataset.nominalAtributos)
        clasificador.datosTrain = copia.datos
      else:
        clasificador.datosTrain = train

      clases = clasificador.clasifica(test,train, dataset.diccionarios, dataset.nominalAtributos)
      aux.append(clasificador.error(test,clases))
    return aux

  def entrenamiento(self, train):
    self.calcularMedDesv(train)
    pass

  # Se calcula la media y la desv tipica para luego normalizar
  def calcularMedDesv(self, train):
    for i in range(len(train[0]-1)):
      suma = np.sum(train[:,i])
      aux=[]
      aux.append(suma/len(train[:,i]))
      aux.append(np.std(train[:,i]))
      self.listaMatrices.append(aux)
    pass

  # Metodo de normalizacion de datos se le resta al dato la media y se divide entre la desv tipica
  def normalizarDatos(self,datos, atributosDiscretos):
    for i in range(len(datos[0])-1):
      if atributosDiscretos[i] == False:
        datos[:,i]= (datos[:,i] - self.listaMatrices[i][0])/self.listaMatrices[i][1]
    pass  
 
  # Metodo que clasifica devuelve los N vecinos proximos
  def clasifica(self, test, train, diccionario, atributosDiscretos): 
    distancias=[]
    elem=[]
    
    # COmprobacion para que no salte error si el tamaño de la particion es mayor de lo que debe, ya que imposibilita realizar vecinos
    if(self.k>len(train)):
      print("El numero de vecinos no puede ser mayor al de la particion de train")
    
    # Si se desea normalizar, normaliza
    if norm:
      self.normalizarDatos(test, atributosDiscretos)

    #para cada dato se calcula su distancia con el resto y se introduce en un array de distancias
    for datosTest in test:
      distancias = []
              
      for datosTrain in train:
        d = 0
        for i in range(len(datosTest)-1):
            d += (datosTest[i]-datosTrain[i])**2
        distEuclidea = math.sqrt(d)
        aux=[]
        aux.append(distEuclidea)
        aux.append(datosTrain[-1])
        distancias.append(aux)

      
      sortedD=sorted(distancias)
      clase=[]

      # Se coge las K (num vecinos) mas proximas y se devuelve en un numpy array
      for j in range(self.k):
        clase.append(sortedD[j][1])
      clase = collections.Counter(clase)
      elem.append(clase.most_common()[0][0])
    return np.array(elem)        



#############################################################################################
#CLASIFICADOR REGRESION LOGISTICA

class ClasificadorRegresionLogistica(Clasificador): 
  litaW=[]
  listaR=[]
  epocas=None
  constAprend = 1

  #Constructor de regresion logistica, nepocas es el numero de epocas que introduzcamos y cons es la constante de aprendizaje por defecto a 1
  def __init__(self,nepocas, cons=1):
    self.listaA=[]
    self.epocas= nepocas
    self.constAprend=cons

  # Metodo de entrenamiento
  def entrenamiento(self,datostrain, norm=True):
    auxW = []

    #Generamos 3 numeros entre -0.5 y 0.5
    for i in range(3):
      numRandom = np.random.uniform(-0.5, 0.5)
      auxW.append(numRandom)

    # Calculamos la sigmoidal para n epocas para cada dato de train
    for i in range(self.epocas):
      for train in datostrain:
        t = np.array([1, train[0], train[1]])
        a = sum(auxW*t)

        if (-a > 700):
          r=0.0
        else:
          r = 1 / (1 + math.exp(-a))

        auxW = auxW -self.constAprend*(r-train[-1])*t
    self.listaW=auxW
    return auxW

  # Metodo de clasificacion
  def clasifica(self, test, train, tipoAtributos, diccionario):
    aux =[]

    #calculamos la sigmoidal para todos los daots de test
    for datosTest in test:
      elem = np.array([1, datosTest[0], datosTest[1]])
      a = sum(elem*self.listaW)

      if (-a > 700):
        r=0.0
      else:
        r = 1 / (1 + math.exp(-a))      

      #si es >0.5 lo introducimos como Clase 1 si no como clase 0 y se devuelve el array de las clases
      if r > 0.5:
        aux.append(1)
      else:
        aux.append(0)
    return np.array(aux)