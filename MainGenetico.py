from Datos import Datos
import EstrategiaParticionado
import ClasificadorGenetico
import numpy as np
import ClasificadorAG 


data = 'ConjuntosDatos/wdbc.data'
dataset=Datos(data)
clasificador=ClasificadorGenetico.ClasificadorGenetico()
cla = ClasificadorAG.ClasificadorAG(1, 10)
a =cla.procesamiento(dataset, clasificador, True)
