# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from ClasificadorScity import ValidacionSimple
from ClasificadorScity import ValidacionCruzada

print(ValidacionSimple('tic-tac-toe.data'))
print()
print(ValidacionCruzada('tic-tac-toe.data',0,3))
