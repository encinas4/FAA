from Datos import Datos
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

###############################################################################################
#SCIKIT LEARN NAIVE BAYES

def NaiveBayesValidacionSimple(fichero,laplace=0.1):
	dataset=Datos(fichero)
	encAtributos =preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
	x = encAtributos.fit_transform(dataset.datos[:,:-1])
	y = dataset.datos[:,-1] 
	xtrain, xtest, ytrain, ytest = train_test_split(x,y)
	clf = naive_bayes.MultinomialNB(alpha=laplace, fit_prior=True)
	clf.fit(xtrain, ytrain)
    
	pred = clf.predict(xtest)
	score = clf.score(xtest, ytest)

	return pred, score


def NaiveBayesValidacionCruzada(fichero, laplace=0.1, part=3):
	dataset=Datos(fichero)
	encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
	x = encAtributos.fit_transform(dataset.datos[:,:-1])
	y = dataset.datos[:,-1] 
	clf = naive_bayes.MultinomialNB(alpha=laplace, fit_prior=True)
	clf = naive_bayes.GaussianNB()
    
	score = cross_val_score(clf, x, y, cv = part)
	pred = cross_val_predict(clf, x, y, cv = part)
	return pred, score

###############################################################################################
#SCIKIT LEARN VECINOS PROXIMOS

def VecinosProximos(fichero, K, particionado, seed=None):
	aux = []
	dataset=Datos(fichero)
	particionado.creaParticiones(dataset, seed)

	for i in range(particionado.numParticiones):
	  train = dataset.extraeDatos(particionado.listaPartic[i].indicesTrain)
	  test = dataset.extraeDatos(particionado.listaPartic[i].indicesTest)
	  KNeightbors = KNeighborsClassifier(K)
	  KNeightbors.fit(train[:,:-1],train[:,-1])
	  aux.append(1-(KNeightbors.score(test[:,:-1],test[:,-1])))
	return aux

###############################################################################################
#SCIKIT LEARN REGRESION LOGISTICA

def RegresionLogistica(fichero, K, particionado, seed=None):
	aux = []
	dataset=Datos(fichero)
	particionado.creaParticiones(dataset, seed)

	for i in range(particionado.numParticiones):
		train = dataset.extraeDatos(particionado.listaPartic[i].indicesTrain)
		test = dataset.extraeDatos(particionado.listaPartic[i].indicesTest)
		res = LogisticRegression()
		res.fit(train[:,:-1],train[:,-1])
		aux.append(1-(res.score(test[:,:-1],test[:,-1])))
	return aux