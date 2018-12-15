from Datos import Datos
import EstrategiaParticionado
import Clasificador
import numpy as np
import ClasificadorAG 


data = 'ConjuntosDatos/wdbc.data'
dataset=Datos(data)
clasificador=Clasificador.ClasificadorGenetico(10)
cla = ClasificadorAG.ClasificadorAG(1, 10)
a =cla.procesamiento(dataset, clasificador, False)
