import os
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
#From the internet
#https://stackoverflow.com/questions/19587118/iterating-through-directories-with-python
swag=[0]
yolo=[0]
blazeit=[0]
finalkl=[0]*30
#insert dir here
algo="PCKLlog"
rootDir = 'C:/Users/Lammy/Desktop/Folderr'
for i in range(30):
    df=np.array(pd.read_csv('C:/Users/Lammy/Desktop/SprinklePC/%i/%s estimatorloss.csv' %(i+1, algo)))
    print("%i: %f" %(i, df.mean()))

#for subdir,dirs,files in os.walk(rootDir):
#    for file in files:
#        if file == "PCKL estimatorloss.csv":
#            df=pd.read_csv('file')
#            #f = open(os.path.join(subdir, file),'r')
#            print(subdir)
#            print(df.mean())
            #for l in f:
        #        l = l.strip()
        #        print(l)
