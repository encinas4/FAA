from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np
import ClasificadorAG 


data = 'ConjuntosDatos/wdbc.data'
dataset=Datos(data)
clasificador=Clasificador.ClasificadorRegresionLogistica(10)
cla = ClasificadorAG.ClasificadorAG(5, 40)
cla.procesamiento(dataset, clasificador)
