from Datos import Datos
import EstrategiaParticionado
import ClasificadorGenetico
import numpy as np
import ClasificadorAG 


data = 'ConjuntosDatos/example1.data'
dataset=Datos(data)
clasificador=ClasificadorGenetico.ClasificadorGenetico()
cla = ClasificadorAG.ClasificadorAG(1, 20)
a =cla.procesamiento(dataset, clasificador, False)