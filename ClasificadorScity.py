
from Datos import Datos
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


def ValidacionSimple(fichero,laplace=0):
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


def ValidacionCruzada(fichero, laplace, part=3):
	dataset=Datos(fichero)
	encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
	x = encAtributos.fit_transform(dataset.datos[:,:-1])
	y = dataset.datos[:,-1] 
	clf = naive_bayes.MultinomialNB(alpha=laplace, fit_prior=True)
	clf = naive_bayes.GaussianNB()
    
	score = cross_val_score(clf, x, y, cv = part)
	pred = cross_val_predict(clf, x, y, cv = part)
	return pred, score

