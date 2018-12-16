import EstrategiaParticionado
from Datos import Datos
import copy
import numpy as np
import random
import math


class ClasificadorAG():
  poblacion = 0;
  nepocas =0;

  def __init__(self, n, p):
    self.nepocas = n
    self.poblacion=p
    pass

  # Metodo que crea reglas de tamano tam, si la representacion es binaria b valdra true
  # y k no sera relevante. Si la representacion es entera b valdra false y se usara k
  # para la generacion de la regla 
  def crearRegla(self,tam, k, b):
    # Para representacion entera
    # generamos aleatorios entre 1 y K para los n "bits" (tam) de la regla
    if(not b):	 # si la representacion es entera
        auxL=[]
        for i in range(tam):
            auxL = np.append(auxL, random.randint(0,k))		#Se introduce un intervalo de 1 a k intervalos
        if random.random()<0.5: # de forma aleatoria generamos 0 o 1 para la clase
            auxL= np.append(auxL,1)
        else:
            auxL= np.append(auxL,0)   		

    #generamos aleatorios 0 y 1 para los n "bits" (tam) de la regla
    else:		# Si la representacion es binaria
        auxL=[]
        for i in range(tam):
            r=[]
            for j in range(k):
                if random.random()<0.5:		# de forma aleatoria generamos 0 o 1 y lo introducimos a la regla
                    r= np.append(r,1)
                else:
                    r= np.append(r,0)
            auxL.append(r)
        if random.random()<0.5:     # Introducimos la clase
            auxL.append(1)
        else:
            auxL.append(0)

    return auxL

  def fitnessPob1(self, auxP,dataset,clasificador, b,k):
    l = []
    for p in auxP:
      estrategia = EstrategiaParticionado.ValidacionCruzada(1)
      estrategia.creaParticiones(dataset, None)
      train = dataset.extraeDatos(estrategia.listaPartic[0].indicesTest)
      f = clasificador.validacion(p, dataset, train, b,k)
      l.append([1-f, p])
    return l
  # Metodo donde se crea la poblacion inicial, si el campo binary es true se hara para representacion binaria
  # si no se aplicara representacion entera.
  def procesamiento(self, dataset, clasificador, binary):
    poblacion=[]
    auxP = []
    tam = len(dataset.nombreAtributos)

    #calculamos el numero de intervalor (caso entero)
    n = self.poblacion	# El numero de individuos es el que le has introducido
    k = int(1+3.322*np.log10(n))
    
    for i in range(n):	#duda la poblacion inicial es de tantos individuos como datos del dataset, creo que es el tamano de train
      individuo=[]
      r = random.randint(1, 5)	# Numero de reglas del individuo que se crea, minimo 1, maximo 5
      for j in range(r):	# Creamos j reglas y las vamos introduciendo una a una en individuo
        individuo.append(self.crearRegla(tam-1,k, binary))
      auxP.append(individuo)
      # En este punto ya tenemos la poblacion inicial creada

    poblacion = self.fitnessPob1(auxP,dataset,clasificador, binary, k)

    for i in range(self.nepocas):
      #Realizamos la seleccion de progenitores
      pobAux = self.seleccionProgenitores(poblacion)
      print("\nProg: ", pobAux)

      #Realixamos el cruce uniforme
      pobAux = self.cruceUniformePob(pobAux)
      #print("\nCruce : ", pobAux)
    
      #realizamos la mutacion
      pobAux = self.mutacionPob(pobAux,binary)
      print("\nMut : ", pobAux)
      #print()
      
      #realizamos el fitness
      #pobAux = self.fitnessPob(self.binariosFnc(pobAux),dataset,clasificador)
      #print("\nFit2 : ", pobAux)

      #Cogemos los mejores
      #poblacion = self.seleccionSup(pobAux,poblacion)
      #print("\nProblacion final: ", poblacion)

    return poblacion[0][0], poblacion[0][1]
    

  def seleccionProgenitores(self, poblacion):
    auxp = sorted(poblacion,key=lambda t: t[0], reverse=True)
    t = int(len(poblacion)/2)
    return auxp[0:t]

  def fitnessPob(self,binarios,dataset,clasificador):
    estrategia = EstrategiaParticionado.ValidacionCruzada(1)
    tam = len(binarios)
    dataSetAux = Datos('ConjuntosDatos/wdbc.data')
    estrategia.creaParticiones(dataset, None)
    train = dataset.extraeDatos(estrategia.listaPartic[0].indicesTest)

    e=[]
    for c in binarios:
      colNum = np.flatnonzero(c)

      dataSetAux.datos = dataset.extraeDatosRelevantes(colNum)
      dataSetAux.diccionarios = dataset.diccionarioRelevante(colNum)
      dataSetAux.nominalAtributos = dataset.atribDiscretosRelevantes(colNum)
      f=(1-clasificador.validacion(estrategia, dataSetAux, train, clasificador)[0])
      #print(f)
      e.append([f, c])
    #np.flatnonzero(e)
    return sorted(e,key=lambda t: t[0], reverse=True)


  def cruceUniformePob(self, pob):
    print(int(len(pob)/2))
    p = []
    for l in range(int(len(pob)/2)):
      p.append(pob[l*2][1])
      p.append(pob[l*2+1][1])
      ptoCruce1 = random.randint(0, len(pob[l*2][1])-1) #elijes el punto de cruce del primer individuo
      ptoCruce2 = random.randint(0, len(pob[(l*2)+1][1])-1) # elije el punto de cruce del segundo individuo

      hijo1 = []
      hijo2 = []
      # cruce en cualquier punto de los dos individuos
      print("pto1", ptoCruce1, "pto2", ptoCruce2, "individuo1", l*2, "individuo2", l*2+1)
      hijo1 = pob[l*2][1][:ptoCruce1]
      hijo1.append(pob[(l*2+1)][1][ptoCruce2:])
      hijo2 = pob[(l*2+1)][1][:ptoCruce2]
      hijo2.append(pob[l*2][1][ptoCruce1:])

      p.append(hijo1)
      p.append(hijo2)

    return p

  def mutacionPob(self, pob, b):
    for l in range(int(len(pob))):
      for i in range(len(pob[l])):#cambiar
        if(not b):
          if random.random() < 0.001:
            print("mutacion")
            pob[l][i] = random.randint(0,k)
        else:
          for j in len(pob[l][i]):
            if random.random() < 0.001:
              pob[l][i][j] = 0 if pob[l][i][j]==1 else 1
    return pob

  def binariosFnc(self, pob):
    ll=[]
    for l in range(len(pob)):
      ll.append(pob[l][1])
    return ll

  def seleccionSup(self,pobAux,poblacion):
    aux = []
    tamElite = int(math.ceil(0.05 * self.p))
    aux.extend(poblacion[0:tamElite])
    aux.extend(pobAux[0:(self.p-tamElite)])
    return sorted(aux,key=lambda t: t[0], reverse=True)