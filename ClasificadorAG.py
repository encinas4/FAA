import EstrategiaParticionado
from Datos import Datos
import copy
import numpy as np


class ClasificadorAG():
  poblacion = 0;
  nepocas =0;

  def __init__(self, n, p):
    self.nepocas = n
    self.p=p
    pass

  def procesamiento(self, dataset, clasificador):
    tam = len(dataset.nombreAtributos)
    intAle = np.random.randint(low=1,high=tam, size=(self.p,1),dtype=np.uint64)
    binarios = np.array([np.hstack((np.ones(n, dtype=np.uint64), np.zeros(tam - n - 1, dtype=np.uint64))) for n in intAle])
    binariosAle = np.array(binarios)

    map(lambda x: np.random.shuffle(x),binariosAle)

    poblacion = self.fitnessPob(binariosAle,dataset,clasificador)
    for i in range(self.p):
      #if max(poblacion[0][1]>0.95):
       # break

      pobAux = self.seleccionProgenitores(poblacion)
      pobAux = self.cruceUniformePob(pobAux)
      pobAux = self.mutacionPob(pobAux)
      pobAux = self.fitnessPob(pobAux,dataset,clasificador)
      poblacion = self.seleccionSup(pobAux,poblacion)

    return poblacion[0][0], np.flatnonzero(poblacion[0][1])

  def seleccionProgenitores(self, poblacion):
   #fitness
   break




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
      print(f)
      e.append([f, c])
    #np.flatnonzero(e)
    return sorted(e,key=lambda t: t[0], reverse=True)


    def cruceUniformePob(self, pob):
      for i in xrange(len(pob)):
        if random.random() < 0.6:
          pob[0:2][i], pob[1:2][i] = pob[1:2][i], pob[0:2][i]
      return pob

    def mutacionPob(self, pob):
      for i in range(len(pob)):
          if random.random() < 0.001:
              p[i] = 0 if p[i]==1 else 1
      return pob

    def seleccionSup(self,pobAux,poblacion):
      aux = []
      tamElite = int(math.ceil(0.05 * self.p))
      aux.extend(poblacion[0:tamElite])
      aux.extend(pobAux[0:(self.tamPob-tamElite)])
      return sorted(aux,key=lambda t: t[1], reverse=True)