# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:23:59 2017

@author: profesores faa
"""

from ClasificadorScity import ValidacionSimple
from ClasificadorScity import ValidacionCruzada

ValidacionSimple('balloons.data')
ValidacionCruzada('balloons.data',0,3)
