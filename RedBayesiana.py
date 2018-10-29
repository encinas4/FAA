


class MatrizDeConfusion():
  TPR=[]
  FNR=[]
  FPR=[]
  TNR=[]


  def _init_(self):
    self.TPR=[]
    self.FNR=[]
    self.FPR=[]
    self.TNR=[]

  def calcular(self, matrix):
    for m in matrix:
      print(m[0,0]/(m[0,0]+m[1,0]))
      self.TPR.append(m[0,0]/(m[0,0]+m[1,0]))
      self.FNR.append(m[1,0]/(m[0,0]+m[1,0]))
      self.FPR.append(m[0,1]/(m[0,1]+m[1,1]))
      self.TNR.append(m[1,1]/(m[0,1]+m[1,1]))
		
 