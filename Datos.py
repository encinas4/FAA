import numpy as np
import collections as collections

class Datos(object):
  
	TiposDeAtributos=('Continuo','Nominal')	
	
 
  # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y auxDic
	def __init__(self, nombreFichero):
		with open(nombreFichero, "r") as f:
			# Numero de filas del conjunto de datos
			self.numFilas=int(f.readline().rstrip())

			# Lista con el nombre de los atributos
			self.nombreAtributos=f.readline().rstrip().split(",")

			# Lista con los tipos de los atributos
			self.tipoAtributos=f.readline().rstrip().split(",")

			# Array con las posiciones de los atributos nominales
			posiciones=[];
			self.numeroAtributos = len(self.nombreAtributos)

			self.nominalAtributos=[]

			# Matriz en la que guardaremos los datos
			datos = np.empty([int(self.numFilas),self.numeroAtributos], dtype=float)

			# Creacion del diccionario y el diccionario auxiliar (para ordenar)
			auxDic=[set() for i in range(self.numeroAtributos)]
			diccionarios = [None for i in range(self.numeroAtributos)]

			# Insertamos las posiciones de los atributos nominales
			i=0;
			for x in self.tipoAtributos:
				if x != "Nominal" and x !="Continuo":
					raise Exception(ValueError)

				if x == "Nominal":
					posiciones.append(i);
				i=i+1;

			content = f.read()
			file = content.splitlines()

			# Creamos el diccionario auxiliar con los atributos
			for fila in file:
				i=0
				for x in fila.rstrip().split(","):
					if i in posiciones and x not in auxDic[i]:
						l=set();
						l = auxDic[i]
						l.add(x);
						auxDic[i]= l
					i=i+1

			# Creamos el diccionario ordenando alfabeticamente a partir del auxiliar
			i=0
			for dic in auxDic:
				j=0
				aux ={}
				for clave in sorted(dic):
					aux.update({clave:j})
					j=j+1
				diccionarios[i]=aux
				i=i+1    		

		# Insertamos los atributos en la matriz
		f=0
		for fila in file:
			i=0
			for x in fila.rstrip().split(","):
				if i in posiciones:			# Si es nominal, se convierte
					datos[f][i] = diccionarios[i][x]
				else:						# Si es discreto, se inserta
					datos[f][i] = x
				i=i+1
			f=f+1
		self.datos = datos
		self.diccionarios = diccionarios

		for i in self.tipoAtributos:
			if i == "Nominal":
				self.nominalAtributos.append(True)
			else:
				self.nominalAtributos.append(False)
	
	# Metodo que extrae datos de la matriz de datos con una lista de indices(posiciones)
	def extraeDatos(self,idx):
		return self.datos[idx,:]

  