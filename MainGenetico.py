from Datos import Datos
import EstrategiaParticionado
import ClasificadorGenetico
import numpy as np
import ClasificadorAG 


data = 'ConjuntosDatos/example1.data'
dataset=Datos(data)
clasificador=ClasificadorGenetico.ClasificadorGenetico()
cla = ClasificadorAG.ClasificadorAG(50, 50)
a =cla.procesamiento(dataset, clasificador, True)
print(a)