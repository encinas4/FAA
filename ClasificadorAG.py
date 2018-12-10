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
    l = dataset.numeroAtributos-1
    poblacion = np.random.randint(2, size=(self.p, l))
    fitness = []
    roulette = []
    for n in range(self.nepocas):


      if(fitness != [] and (max(fitness)>=0.9)):
        return
          
      fitness = []
      roulette = []
      cruzados = []
      siguienteGen = []
      ran = 1
      
      for elemento in poblacion:
        
          dataset_elem = copy.copy(dataset)
          indices = list(np.flatnonzero(elemento))

          dataset_elem.datos = dataset.extraeDatosRelevantes(indices)
          dataset_elem.diccionarios = dataset.diccionarioRelevante(indices)
          dataset_elem.atribDiscretos = dataset.atribDiscretosRelevantes(indices)
          dataset_elem.nombreAtributos = dataset.nombreAtributosRelevantes(indices)
         # print("Los atributos seleccionados son: ",dataset_elem.nombreAtributos[:-1])

          particionado = EstrategiaParticionado.ValidacionCruzada(10)
          fitness.extend(clasificador.validacion(particionado, dataset_elem, clasificador))
#Acierta siempre al clasificar con reg logistica lo cual es muy raro....
          #print"El fitness es: ", 1-fitness[-1]
          #print  fitness
      fitness = list(1-np.array(fitness))

      for i in range(len(fitness)):
          norm = fitness[i]/sum(np.array(fitness))
          roulette.append((poblacion[i], ran-norm, norm))
          ran -= norm
      
      for i in range(int(self.p*0.6)):
          random = np.random.rand()
          for i in range(len(roulette)):
              if ((random > roulette[i][1]) and (random < roulette[i-1][1])):
                  cruzados.append(roulette[i][0])
                  
      for IndiceACruzar in range(1, len(cruzados), 2):
          hijoUno = []
          hijoDos = []
          for subIndice in range(len(cruzados[IndiceACruzar])):
              if (np.random.randint(2) == 0):
                  hijoUno.append(cruzados[IndiceACruzar][subIndice])
                  hijoDos.append(cruzados[IndiceACruzar - 1][subIndice])
              else:
                  hijoUno.append(cruzados[IndiceACruzar - 1][subIndice])
                  hijoDos.append(cruzados[IndiceACruzar][subIndice])
                  
          siguienteGen.append(hijoUno)
          siguienteGen.append(hijoDos)
      
      for i in list(sorted(roulette, key=lambda max:max[2], reverse=True)[0:int(self.p*0.05)]):
          siguienteGen.append(list(i[0]))
      print(siguienteGen)
      
      datosAMutar = []
      numDatosMutacion = int(0.35*len(poblacion))
      
      for i in range(numDatosMutacion):
          random = np.random.randint(len(poblacion))
          datosAMutar.append(poblacion[random])
      
      for valor in datosAMutar:
          for bit in range(len(valor)):
              random = np.random.randint(1000)
              if random < 1:
                  if valor[bit] == 0: 
                      valor[bit] = 1
                  else: 
                      valor[bit] = 0
                                   
      siguienteGen += datosAMutar
      poblacion=np.array(siguienteGen)
      
      print"El maximo de fitness es: ",max(fitness)
      tuple = list(sorted(roulette, key=lambda max:max[2], reverse=True))[0]
      print "Max fitness:", tuple[0]
      print "Ha seleccionado los siguientes atributos: "
      
      valoresSalida = []
      
      for i in range(len(tuple[0])):
          if (tuple[0][i] == 1):
              valoresSalida.append(dataset.nombreAtributos[i])
      print valoresSalida