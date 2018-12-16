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
    self.p=p
    pass
  def crearReglas(self,tam, k, b):
    l = []
    if(not b):
      for i in range(tam):
        auxL=[]
        auxL=np.random.randint(k+1, size=tam)
        if random.random()<  0.5:
          auxL= np.append(auxL,1)
        else:
          auxL= np.append(auxL,0)
        l.append(auxL)
    print(l)
    return l

  def fitnessPob1(self, auxP,dataset,clasificador, b,k):
    l = []
    for p in auxP:
      f = clasificador.evaluar(p, dataset, b,k)
      l.append([1-f, p])
    return l





  def procesamiento(self, dataset, clasificador, binary):
    poblacion=[]

    auxP = []
    tam = len(dataset.nombreAtributos)
    n = len(dataset.datos)
    k = int(1+ 3.322*np.log10(n))
    
    for i in range(n):
      individuo=[]
      r = random.randint(1, 5)
      for j in range r:
        individuo.append(self.crearReglas(tam-1,k, binary))
      auxP.append(individuo)

    poblacion = self.fitnessPob1(auxP,dataset,clasificador, binary, k)




    """
     for i in range(self.n):
      pobAux = self.seleccionProgenitores(poblacion)
      #print("\nProg: ", pobAux)
     
      pobAux = self.cruceUniformePob(pobAux)
      #print("\nCruce : ", pobAux)
    
      pobAux = self.mutacionPob(pobAux)
     # print("\nMut : ", pobAux)
      #print()
      
      pobAux = self.fitnessPob(self.binariosFnc(pobAux),dataset,clasificador)
     # print("\nFit2 : ", pobAux)
      poblacion = self.seleccionSup(pobAux,poblacion)
      #print("\nProblacion final: ", poblacion)

    return poblacion[0][0], poblacion[0][1]

    #print("PP:",  poblacion)
   
    """
    return 1
    

  def seleccionProgenitores(self, poblacion):
    auxp = sorted(poblacion,key=lambda t: t[0], reverse=True)
    t = int(len(poblacion)/2)
    return auxp[0:t]
   




  def fitnessPob(self,binarios,dataset,clasificador):
    estrategia = EstrategiaParticionado.ValidacionSimple(0.7,1)
    tam = len(binarios)
    dataSetAux = Datos('ConjuntosDatos/wdbc.data')

    e=[]
    for c in binarios:
      colNum = np.flatnonzero(c)

      dataSetAux.datos = dataset.extraeDatosRelevantes(colNum)
      dataSetAux.diccionarios = dataset.diccionarioRelevante(colNum)
      dataSetAux.nominalAtributos = dataset.atribDiscretosRelevantes(colNum)
      f=(1-clasificador.validacion(estrategia, dataSetAux, clasificador)[0])
      #print(f)
      e.append([f, c])
    #np.flatnonzero(e)
    return sorted(e,key=lambda t: t[0], reverse=True)


  def cruceUniformePob(self, pob):
    for l in range(int(len(pob)/2)):
      for i in range(len(pob[l][1])):
        if random.random()<  0.2:
          print("YES\n")
          pob[l][1][i], pob[l+1][1][i] = pob[l][1][i], pob[l+1][1][i]
    return pob

  def mutacionPob(self, pob):
    for l in range(int(len(pob))):
      for i in range(len(pob[l][1])):#cambiar
        if random.random() < 0.001:
          pob[l][1][i] = 0 if pob[l][1][i]==1 else 1
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