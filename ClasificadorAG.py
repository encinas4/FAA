import EstrategiaParticionado
from Datos import Datos
import copy
import numpy as np
import random
import math
import matplotlib.pyplot as plt


class ClasificadorAG():
  poblacion = 0;
  nepocas =0;
  medias=[]
  mejores=[]

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

  def fitnessPob1(self, auxP,dataset,train,clasificador, b,k):
    l = []
    for p in auxP:
      f = clasificador.validacion(p, dataset,train, b,k)
      l.append([1-f, p])
    return l
  # Metodo donde se crea la poblacion inicial, si el campo binary es true se hara para representacion binaria
  # si no se aplicara representacion entera.
  def procesamiento(self, dataset, clasificador, binary):
    poblacion=[]
    self.mejores=[]
    self.medias=[]
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
      #print("AuxP", auxP)
    estrategia = EstrategiaParticionado.ValidacionCruzada(1)
    estrategia.creaParticiones(dataset, None)
    train = dataset.extraeDatos(estrategia.listaPartic[0].indicesTest)
    poblacion = self.fitnessPob1(auxP,dataset, train,clasificador, binary, k)
    for i in range(self.nepocas):
      #Realizamos la seleccion de progenitores
      pobAux = self.seleccionProgenitores(poblacion)
      #print("\nProg: ", pobAux)

      #Realixamos el cruce uniforme
      pobAux = self.cruceUniformePob(pobAux)
      #print("\nCruce : ", pobAux)

      #realizamos la mutacion
      pobAux = self.mutacionPob(pobAux,binary,k)
      #print("\nMut : ", pobAux)
      #print()

      #realizamos el fitness
      #print(pobAux, poblacion)
      pobAux = self.fitnessPob1(pobAux,dataset,train,clasificador,binary,k)
      #print("\nFit2 : ", pobAux)

      #Cogemos los mejores
      poblacion = self.seleccionSup(pobAux,poblacion)
      #print("\nProblacion final: ", poblacion)

      listaF = 0
      listaFi = []
      for p in poblacion:
          #print(p)
          listaF+=p[0]
          listaFi.append(p[0])
      media = listaF/len(poblacion)
      self.medias.append(media)
      listaFi.sort()
      self.mejores.append(listaFi[-1])

    estrategia = EstrategiaParticionado.ValidacionCruzada(1)
    estrategia.creaParticiones(dataset, None)
    test = dataset.extraeDatos(estrategia.listaPartic[0].indicesTest)
    pred = clasificador.clasificar(poblacion[0][1], dataset, test, binary,k)
    return clasificador.error(pred, test)


  def imprimirGraficas(self):
      i=0
      for m in self.mejores:
          print("EL mejor de la epoca ", i, "tiene fitness", m)
          i+=1
      plt.plot(self.mejores,"bo", label="mejores fitness")
      plt.plot(self.medias,"r*", label="medias")
      plt.ylabel("valor del fitness")
      plt.xlabel("Numero de epocas")
      plt.legend()
      plt.title("Grafica de los mejores fitnes y media")
      plt.show()


  def seleccionProgenitores(self, poblacion):
    auxp = sorted(poblacion,key=lambda t: t[0], reverse=True)
    t = int(len(poblacion)/2)
    return auxp[0:t]



  def cruceUniformePob(self, pob):
    p = []
    for l in range(int(len(pob)/2)):
      p.append(pob[l*2][1])
      p.append(pob[l*2+1][1])
      ptoCruce1 = random.randint(0, len(pob[l*2][1])-1) #elijes el punto de cruce del primer individuo
      ptoCruce2 = random.randint(0, len(pob[(l*2)+1][1])-1) # elije el punto de cruce del segundo individuo

      hijo1 = []
      hijo2 = []
      # cruce en cualquier punto de los dos individuos
      hijo1[:ptoCruce1] = pob[l*2][1][:ptoCruce1]
      hijo1[ptoCruce1:] =pob[(l*2+1)][1][ptoCruce2:]
      hijo2[:ptoCruce2] = pob[(l*2+1)][1][:ptoCruce2]
      hijo2[ptoCruce2:] = pob[l*2][1][ptoCruce1:]

      p.append(hijo1)
      p.append(hijo2)
    return p

  def mutacionPob(self, pob, b,k):
    if(not b):
      for p in range(len(pob)):
        for l in range(int(len(pob[p]))):
          for i in range(len(pob[p][l])):#cambiar
            if random.random() < 0.001:
              pob[p][l][i] = random.randint(0,k)
    else:
      for p in range(len(pob)):
        for l in range(int(len(pob[p]))):
          for i in range(len(pob[p][l])-1):#cambiar
            for j in range(len(pob[p][l][i])):
              if random.random() < 0.001:
                pob[p][l][i][j] = 0 if pob[p][l][i][j]==1 else 1

    return pob

  def binariosFnc(self, pob):
    ll=[]
    for l in range(len(pob)):
      ll.append(pob[l][1])
    return ll

  def seleccionSup(self,pobAux,poblacion):
    aux = []
    tamElite = int(math.ceil(0.05 * self.poblacion))
    aux.extend(poblacion[0:tamElite])
    aux.extend(pobAux[0:(self.poblacion-tamElite)])
    return sorted(aux,key=lambda t: t[0], reverse=True)
