# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



def testfu(x,a):
    for i in range(int(x)):
        x+=(x-i)/(i*1.+a)
    return x

testfuv=np.vectorize(testfu)

def reihe(n,L):
    out=np.zeros(L)
    for i in range(L):
        b=True
        while b:
            out[i]=np.random.randint(0,n)
            if out[i]!=out[i-1]:
                b=False
    return out

def fibi(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def cumfibi(n):
    out=[]
    a, b = 0, 1
    out.append(a)
    out.append(b)
    for i in range(n):
        a, b = b, a + b
        out.append(b)
    return out


def chaosgamen(r,n,L):
    a=2*np.pi/(n*1.)
    x=[]
    y=[]
    for i in range(n):
        x.append(np.sin(a*i+0.25*np.pi))
        y.append(np.cos(a*i+0.25*np.pi))
    x.append(x[0])   
    y.append(y[0])    
    path=np.random.randint(0,n,[L])
    x1=[]
    y1=[]
    xstart=0
    ystart=0    
    x1.append(xstart)
    y1.append(ystart)
    for i in range(len(path)):
        x1.append(x1[i-1]+r*(x[path[i]]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[path[i]]-y1[i-1]))
    plt.plot(x,y,c='r')
    plt.scatter(x1[:],y1[:],s=1)

def datachaosgamen(r,n,path):
    a=2*np.pi/(n*1.)
    x=[]
    y=[]
    for i in range(n):
        x.append(np.sin(a*i+0.25*np.pi))
        y.append(np.cos(a*i+0.25*np.pi))
    xstart=5
    ystart=5
    x1=[]
    y1=[]
    while abs(xstart)+abs(ystart)>1  :
        xstart=np.random.uniform(-1,1)
        ystart=np.random.uniform(-1,1)
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x1.append(x1[i-1]+r*(x[int(path[i])]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[int(path[i])]-y1[i-1]))
    plt.scatter(x1[:],y1[:],s=1)
    return x1,y1



def condatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)-np.min(path)+adder))
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+r*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[i]-y1[i-1]))
    plt.scatter(x1[:],y1[:],s=1)
    return x1,y1 

def multicondatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)-np.min(path)+adder))

    for k in range(len(r)):
        x=[]
        y=[]
        x1=[]
        y1=[]
        y1.append(0)
        x1.append(0)
        for i in range(len(path)):
            x.append(np.sin(a*path[i]))
            y.append(np.cos(a*path[i]))
            x1.append(x1[i-1]+r[k]*(x[i]-x1[i-1]))
            y1.append(y1[i-1]+r[k]*(y[i]-y1[i-1]))
        plt.scatter(x1[:],y1[:],s=0.5)


#bmulticondatachaosgamen(np.linspace(0,2,50),np.sin(np.arange(0,3000,10,dtype='float64')))
#multicondatachaosgamen(np.linspace(0,1,50),np.sin(np.arange(0,3000,2,dtype='float64'))*np.sin(0.5*np.arange(0,3000,2,dtype='float64')))
#conpdatachaosgamen(0.5,np.sin(np.arange(0,400,1,dtype='float64')))
#bconpdatachaosgamen(2,np.sin(np.arange(0,600,0.8,dtype='float64')))
#multicondatachaosgamen(np.linspace(0.1,1.9,10),(1+np.arange(0,1000,1)*5000*np.sin(np.arange(0,1000,1))))
#condatachaosgamen(2,(1+np.arange(0,6000,1)*5000*np.sin(np.arange(0,6000,1))))
#new2condatachaosgamen((1+np.arange(0,19000,1)*5000*np.sin(np.arange(0,19000,1))))

def bmulticondatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)-np.min(path)+adder))
    #fig = plt.figure()
    #fig.patch.set_facecolor('black')
    ax = plt.gca()
    ax.patch.set_facecolor('black')
    for k in range(len(r)):
        x=[]
        y=[]
        x1=[]
        y1=[]
        y1.append(0)
        x1.append(0)
        for i in range(len(path)):
            x.append(np.sin(a*path[i]))
            y.append(np.cos(a*path[i]))
            x1.append(x1[i-1]+r[k]*(x[i]-x1[i-1]))
            y1.append(y1[i-1]+r[k]*(y[i]-y1[i-1]))
        plt.plot(x1[:],y1[:],'.')


def multiconpdatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)-np.min(path)+adder))

    for k in range(len(r)):
        x=[]
        y=[]
        x1=[]
        y1=[]
        y1.append(0)
        x1.append(0)
        for i in range(len(path)):
            x.append(np.sin(a*path[i]))
            y.append(np.cos(a*path[i]))
            x1.append(x1[i-1]+r[k]*(x[i]-x1[i-1]))
            y1.append(y1[i-1]+r[k]*(y[i]-y1[i-1]))
        plt.plot(x1[:],y1[:])
    

def ncondatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)-np.min(path)+adder))
    print a
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+r*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[i]-y1[i-1]))
    return x1,y1 

def conpdatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)*1.-np.min(path)*1.+adder))
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+r*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[i]-y1[i-1]))
    plt.plot(x1[:],y1[:],'bx-')
    return x1,y1 
#dfdf
#dsf
def bconpdatachaosgamen(r,path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)*1.-np.min(path)*1.+adder))
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+r*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[i]-y1[i-1]))
    plt.plot(x1[:],y1[:],'k')
    return x1,y1 

def newconpdatachaosgamen(path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)*1.-np.min(path)*1.+adder))
    r=1/np.double(len(path))
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+r*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[i]-y1[i-1]))
    plt.plot(x1[:],y1[:],'bx-')
    return x1,y1 

def newcondatachaosgamen(path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)*1.-np.min(path)*1.+adder))
    r=1/np.double(len(path))
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+r*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[i]-y1[i-1]))
    plt.scatter(x1[:],y1[:],s=1)
    return x1,y1 

def new2condatachaosgamen(path):
    adder=np.diff(np.sort(np.unique(path))).min()
    a=2*np.pi/(1*(np.max(path)*1.-np.min(path)*1.+adder))
    #r=1/np.double(len(path))
    x=[]
    y=[]
    x1=[]
    y1=[]
    y1.append(0)
    x1.append(0)
    for i in range(len(path)):
        x.append(np.sin(a*path[i]))
        y.append(np.cos(a*path[i]))
        x1.append(x1[i-1]+1/(1.*len(x1))*(x[i]-x1[i-1]))
        y1.append(y1[i-1]+1/(1.*len(x1))*(y[i]-y1[i-1]))
    plt.scatter(x1[:],y1[:],s=1)
    return x1,y1 


    
def density(x1,y1,xgrid,ygrid):
    x1=np.array(x1)
    y1=np.array(y1)
    out=np.zeros([xgrid,ygrid])
    x_sortindex=np.argsort(x1)
    x1=np.sort(x1)
    y1=y1[x_sortindex]    
    hx=np.histogram(x1,xgrid)
    hy=np.histogram(y1,ygrid)
    counter=0
    for i in range(xgrid):
        for j in range(ygrid):
            for ystep in range(counter,counter+hx[0][i]):
                if (y1[ystep] >= hy[1][j]) and (y1[ystep] < hy[1][j+1]):
                    out[i,j]+=1
        counter=counter+hx[0][i]
    return out        
        
#np.histogram2d(a,b,[256,256])[0]
    

#chaosgamen(1.5,3,20000)

def chaosgame(r):
    x=[-1,1,0]
    y=[0,0,1]
    path=np.random.randint(0,3,[2000])
    xstart=5
    ystart=5
    x1=[]
    y1=[]
    while abs(xstart)+abs(ystart)>1  :
        xstart=np.random.uniform(-1,1)
        ystart=np.random.uniform(0,1)
    x1.append(xstart)
    y1.append(ystart)
    for i in range(len(path)):
        x1.append(x1[i-1]+r*(x[path[i]]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[path[i]]-y1[i-1]))
    plt.scatter(x,y,c='r')
    plt.scatter(x1,y1)
    
#chaosgame(0.5)


def fac(k):
    if k%1!=0:
        print("gerundet")
        k=k-k%1
    return np.cumprod(np.arange(1,k+1))[-1]
