# -*- coding: utf-8 -*-

import numpy as np
import scipy
import Image
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/kernke/Schreibtisch/Python/Combi/mountsavgol')
from mountsavgol import savgolmount as m
from scipy.signal import argrelmax

test=[0,1,1,2,0,0]
test2=[0,3,2,1,0,0]
a=np.random.randint(0,10000,100)
x=np.arange(100)
b=np.exp(-(x/50.)**2)*10000+x**2
#del c
#c=np.correlate(b+a,b+a,mode='full')#mode='full')
#plt.plot((b+a)/np.max(b+a),'b')
#plt.plot(c[int(len(c)/2)+2:]/np.max(c),'g')
#plt.plot(b/np.max(b),'r')

path='/home/kernke/Schreibtisch/Convert/Run1C07/'
files=os.listdir(path)
files=np.sort(files)

bild=np.array(Image.open(path+files[68]))
sbild=m.smooth(bild)
arr=np.array([[0,2,0],[0,2,0],[1,0,3],[-3,1,5],[4,2,3]])

def descend(relmaximag,imag):
    out=np.zeros([np.sum(relmaximag!=0*1.),3])
    count=0
    for i in range(len(relmaximag[:,0])):
        for j in range(len(relmaximag[0,:])):
            if relmaximag[i,j]!=0:
                out[count,0]=i
                out[count,1]=j
                out[count,2]=imag[i,j]
                count+=1
    sorter=np.argsort(out[:,2])
    return out[sorter[::-1],0:2]

def fastdescend(relmaximag,imag):
    '''relamximag breuecksichtigte Pixel nullen und einsen
    imag bild
    returns von oben abfallende (pixelwert) serie der pixelpositionen'''
    def smallfunc(i,j,mask,pic):
        if mask[i,j]!=0:
            return i,j,pic[i,j]        
    smallfuncv=np.vectorize(smallfunc,excluded=['mask','pic'])

    out=np.zeros([np.sum(relmaximag!=0*1.),3])
    in1=np.repeat(np.arange(len(imag[:,0])),len(imag[0,:]))
    in2=np.tile(np.arange(len(imag[0,:])),len(imag[:,0]))
    out=smallfuncv(i=in1,j=in2,mask=relmaximag,pic=imag)
    out=np.array(out) 
    sorter=np.argsort(out[2,:])
    return np.transpose(out[0:2,sorter[::-1]]).astype(int)

                

def corpic(image):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    #a=m.smooth(a,n=21)
    corx=[]
    cory=[]
    for i in range(form[0]-1):
        corx.append(np.correlate(a[:,i],a[:,i+1]))
    for i in range(form[1]-1):
        cory.append(np.correlate(a[i,:],a[i+1,:]))
        #cory.append(np.sum(a[i,:]*a[i+1,:]))
    return corx,cory

def corpiccor(image):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    #a=m.smooth(a,n=21)
    corx=[]
    cory=[]
    for i in range(form[0]-1):
        corx.append(np.corrcoef(a[:,i],a[:,i+1])[0,1])
    for i in range(form[1]-1):
        cory.append(np.corrcoef(a[i,:],a[i+1,:])[0,1])
    return corx,cory

    
def corpicsub(image):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    a=m.smooth(a,n=21)
    corx=[]
    cory=[]
    for i in range(form[0]-1):
        corx.append(np.sum(np.abs(a[:,i]-a[:,i+1])))
    for i in range(form[1]-1):
        cory.append(np.sum(np.abs(a[i,:]-a[i+1,:])))
    return corx,cory

def corpicsub2(image):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    a=m.smooth(a)
    corx=[]
    cory=[]
    for i in range(form[0]-2):
        corx.append(np.sum((a[:,i]-a[:,i+1])*(a[:,i+1]-a[:,i+2])))
    for i in range(form[1]-2):
        cory.append(np.sum((a[i,:]-a[i+1,:])*(a[i,:]-a[i+2,:])))
    return corx,cory


def corpicq(image):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    a=m.smooth(a,n=21)
    corx=[]
    cory=[]
    for i in range(form[0]-1):
        corx.append(np.correlate(a[:,i],a[:,i+1])**2)
    for i in range(form[1]-1):
        cory.append(np.correlate(a[i,:],a[i+1,:])**2)
        #cory.append(np.sum(a[i,:]*a[i+1,:]))
    return corx,cory

def corpicul(image):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    a=m.smooth(a,n=21)
    corx=[]
    cory=[]
    for i in range(form[0]-1):
        corx.append(np.correlate(a[:,i],a[:,i+1],mode='same'))
    for i in range(form[1]-1):
        cory.append(np.correlate(a[i,:],a[i+1,:],mode='same'))
        #cory.append(np.sum(a[i,:]*a[i+1,:]))
    return corx,cory
