    ###################################################################################

class ClasificadorRegresionLogistica(Clasificador): 
  litaW=[]
  listaR=[]
  epocas=None
  def _init_(self,nepocas):
    self.listaA=[]
    self.epocas= nepocas

  # TODO: implementar
  def entrenamiento(self,datostrain,constAprend):
    w = np.random.uniform(low=-0.5, high=0.5, size=(len(train),))
    for i in range(constAprend):
      for train in datostrain:
        res = w*train
        r=1/1+ math.exp(np.sum(res))
        w= w - self.epocas*(r-train[-1])*train
    self.listaW = w

  def clasifica(self, test):
    aux =[]
    for datosTest in test:
      elem = np.array(1, datosTest[0], datosTest[1])
      a = elem*self.w
      r = 1/(1+math.exp(a))
      if r > 0.5:
        aux.append(1)
      else:
        aux.append(0)
    return aux

  def validacion(self,particionado,dataset,clasificador, laplace=0 , seed=None, constAprend = 1):
    aux = []
    particionado.creaParticiones(dataset, seed)

    for i in range(particionado.numParticiones):
      train = dataset.extraeDatos(particionado.listaPartic[i].indicesTrain)
      test = dataset.extraeDatos(particionado.listaPartic[i].indicesTest)
      clasificador.entrenamiento(train, constAprend)
      clases = clasificador.clasifica(test)
      aux.append(clasificador.error(test, clases))
    return aux