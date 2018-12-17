from Datos import Datos
import EstrategiaParticionado
import ClasificadorGenetico
import numpy as np
import ClasificadorAG 

data = 'ConjuntosDatos/wdbc.data'
dataset=Datos(data)
clasificador=ClasificadorGenetico.ClasificadorGenetico()
cla = ClasificadorAG.ClasificadorAG(10, 100)
a =cla.procesamiento(dataset, clasificador, True)
print(a)