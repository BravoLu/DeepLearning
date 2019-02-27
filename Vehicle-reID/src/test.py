import numpy as np


#a = np.arange(1,2,0.5)
#b = a[:,np.newaxis]
#c = [[1.2,1.0],
#      [1.5,1.5]]
#d = (c == b) 
#print(b)
#print(d)
    
matches = np.asarray([1,0,1,0,2,0,0])
valid = [True,True,True,True,False,False,False]
print(matches[valid])
index = np.nonzero(matches[valid])[0]
print(index)
