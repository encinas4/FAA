import EstrategiaParticionado
import Datos
import numpy as np
import copy

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
    for i in range(p):
      if max(poblacion[0][1]>0.95):
        break

      pobAux = self.seleccionProgenitores(poblacion)
      pobAux = self.cruceUniformePob(pobAux)
      pobAux = self.mutacionPob(pobAux)
      pobAux = self.fitnessPob(pobAux,dataset,clasificador)
      poblacion = self.seleccionSup(pobAux,poblacion)



  def fitnessPob(self,binarios,dataset,clasificador):
    estrategia = EstrategiaParticionado.ValidacionSimple(porcentajeTrain=0.7)
    tam = len(binarios)
    #l=Parallel(n_jobs=-1)(delayed(unwrap_self_fit)(p) for p in  izip([self] * tam, binarios, [dataset] * tam,[clasificador] * tam, [estrategia]*tam))
    dataSetAux = Datos()
    e=[]
    c=[]

    for c in binario:
      colNum = np.flatnonzero(binarios)
      dataSetAux.datos = dataset.extraeDatosRelevantes(colNum)
      dataSetAux.diccionarios = dataset.diccionarioRelevante(colNum)
      dataSetAux.nominalAtributos = dataset.atribDiscretosRelevantes(colNum)
      e.extend(clasificador.validacion(estrategia, dataSetAux, clasificador))
      c.extend(col)
      #col y error que habria que ordenar?
    return np.flatnonzero(poblacion[0][0]), poblacion[0][1]
      


    return sorted(l,key=lambda t: t[1], reverse=True)


    def cruceUniformePob(self, pob):
      for pp in pob:
        for i in xrange(len(p)):
            if random.random() < 0.6:
                pp[0:2][i], pp[1:2][i] = pp[1:2][i], pp[0:2][i]
      return pob
    def mutacionPob(self, pob):
      for p in pob:
        for i in range(len(p)):
            if random.random() < 0.001:
                p[i] = 0 if p[i]==1 else 1

    def seleccionSup(self,pobAux,poblacion):
      aux = []
      tamElite = int(math.ceil(0.05 * self.p))
      aux.extend(poblacion[0:tamElite])
      aux.extend(pobAux[0:(self.tamPob-tamElite)])
      return sorted(aux,key=lambda t: t[1], reverse=True)