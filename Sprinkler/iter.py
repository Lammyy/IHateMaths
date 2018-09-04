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
algo="PCADVR"
rootDir = 'C:/Users/Lammy/Desktop/Folderr'
for i in range(30):
    if i!=5:
        if i!=13:
    #        if i!=17:
            df=np.array(pd.read_csv('C:/Users/Lammy/Desktop/Folderr/%i/%s estimatorloss.csv' %(i+1, algo)))
            swag=swag+df
            #print(swag)
            gh=np.array(pd.read_csv('C:/Users/Lammy/Desktop/Folderr/%i/%s nelbo.csv' %(i+1, algo)))
            yolo=yolo+gh

            kl=np.array(pd.read_csv('C:/Users/Lammy/Desktop/Folderr/%i/%s truklavg.csv' %(i+1, algo)))
            finalkl[i]=kl[-1]
            blazeit=blazeit+kl
print(finalkl)
finalkl=np.delete(finalkl,[5,13])
print(finalkl)
print(np.mean(finalkl))
print(np.std(finalkl))
yolo=yolo/28
swag=swag/28
blazeit=blazeit/28
plt.plot(swag)
plt.savefig("C:/Users/Lammy/Desktop/estimator losses/%s estimatorloss" %(algo))
plt.close()
plt.plot(yolo)
plt.savefig("C:/Users/Lammy/Desktop/nelbos/%s nelbo" %(algo))
plt.close()
plt.plot(blazeit)
plt.savefig("C:/Users/Lammy/Desktop/truklmins/%s truklavg" %(algo))
plt.close()
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
