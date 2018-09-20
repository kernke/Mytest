# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
    

def mountfast(image,lim=0):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image
    form=list(np.shape(a))
    if lim==0:
        lim=np.sum(form)
    
    oldm=np.ones(form)
    check=np.zeros(form)
    count=0
    nn=np.zeros([4]+form)
    nn2=np.zeros([4]+form)

    nn2[0,:,:-2]=a[:,2:]
    nn2[0,:,-2:]=-np.inf
    nn2[1,:,2:]=a[:,0:-2]
    nn2[1,:,:2]=-np.inf
    nn2[2,2:,:]=a[0:-2,:]
    nn2[2,:2,:]=-np.inf    
    nn2[3,:-2,:]=a[2:,:]
    nn2[3,-2:,:]=-np.inf


    nn[0,:,0:-1]=a[:,1:]
    nn[0,:,-1]=-np.inf
    nn[1,:,1:]=a[:,0:-1]
    nn[1,:,0]=-np.inf
    nn[2,1:,:]=a[0:-1,:]
    nn[2,0,:]=-np.inf    
    nn[3,0:-1,:]=a[1:,:]
    nn[3,-1,:]=-np.inf
    for i in range(4):
        nn[i,:,:]=(nn[i,:,:]-a)>0
        nn2[i,:,:]=(nn2[i,:,:]-a)>0
    nn=((nn2+nn)>0)*1.    
    #nn next neighbour difference
    #bo=np.zeros([4]+form)
    #for i in range(4):
    #    bo[i,:,:]=nn[i,:,:]>0#bo is 1 whenever slope positive otherwise zero
        #so in contrast to a gradient all positive slopes are taken into account
    tei=np.sum(nn,0)
    #rm relatvie maxima (and completey flat points)
    rm= tei==np.zeros(form)

    for i in range(len(tei[:,0])):
        for j in range(len(tei[0,:])):
            if tei[i,j]==0:
                tei[i,j]=np.inf
    check=np.zeros(form)
    rest=0
    newm=np.zeros(form)
    bn=np.zeros([4]+form)
    cn=np.zeros([4]+form)
    fm=np.zeros(form)
    while count<lim:     
        #oldm+=inspectors/4.-fm
        
        mehr=oldm/tei
        for i in range(4):
            bn[i,:,:]=nn[i,:,:]*mehr
        #bn contains how many climbers leave a point in every direction
        cn[1,:,0:-1]=bn[1,:,1:]
        cn[0,:,1:]=bn[0,:,0:-1]
        cn[3,1:,:]=bn[3,0:-1,:]
        cn[2,0:-1,:]=bn[2,1:,:]
        #cn contain how many climbers arrive from every direction
        rr=np.sum(np.abs(check-oldm))
        
        for i in range(4):
            newm+=cn[i,:,:]

        check=np.copy(oldm)
        fm=rm*oldm
        oldm=newm+fm
        if np.isclose(rr,0):
            print('loops:')
            print(count)
            break
        else:
            #rest=rr
            count+=1
        '''    
        inspectors[:,:]=0
        inspectors[1:,:]+=fm[:-1,:]
        inspectors[:-1,:]+=fm[1:,:]
        inspectors[:,1:]+=fm[:,:-1]
        inspectors[:,:-1]+=fm[:,1:]
        '''
        newm[:,:]=0

    print('Anzahl relativer Maxima: '+str(np.sum(rm)))
    print('Restdifferenz: '+str(100.*rest/np.double(form[0]*form[1]))+' %')
    return oldm


def find_neighbours_in_dist(x,y,distan):
    xy=np.array([x,y]).T    
    dist=distance.cdist(xy,xy,"euclidean")
    distl=np.tril(dist)
    neigh1,neigh2=np.where(np.isclose(distl,distan))
    neighbours=[]
    checker=[]
    for i in range(len(neigh1)):
        if i in checker:
            pass
        else:
            nei=[]
            nei.append(neigh1[i])
            nei.append(neigh2[i])
            for j in range(i+1,len(neigh1)):
                if neigh1[j] in nei:
                    nei.append(neigh2[j])
                    checker.append(j)
                elif neigh2[j] in nei:
                    nei.append(neigh1[j])
                    checker.append(j)
            neighbours.append(nei)
    return neighbours


#if 'thr' in locals():
def flood(image,floodpoints,lim=0,floodthreshold=1,clustercheckradius=0):
    if isinstance(image, basestring):
        a=np.array(Image.open(image))
    else:
        a=image        
    
    form=list(np.shape(a))
    maxi=np.zeros(form)
    
    if lim==0:
        lim=np.sum(form)
    
    if np.shape(floodpoints)==np.shape(a):
        bxy=np.unravel_index(np.argsort(floodpoints,axis=None)[::-1],np.shape(a))
        bl=np.vstack((bxy[0],bxy[1])).T
        for i in range(len(bl)):
            if floodpoints[bl[i][0],bl[i][1]]<floodthreshold:
                thr=i
                break
        b=bl[:thr]
        val=np.sort(np.reshape(floodpoints,form[0]*form[1]))[::-1][:thr]
        maxi=(floodpoints>floodthreshold)*floodpoints
        if thr==0:
            print 'No point bigger than the floodthreshold'
    
    else:
        b=floodpoints
        val=a[b[1,:],b[0,:]]
        for i in range(len(b)):
            maxi[b[i][0],b[i][1]]=a[b[i][0],b[i][1]]
        maxi=(maxi>floodthreshold)*maxi
    
    '''    
    maxi=np.zeros(form)
    for i in range(len(b)):
        maxi[b[i][0],b[i][1]]=len(b)-i+1
    '''
    if clustercheckradius>0:
        near=[]
        vald=np.sqrt(val)*clustercheckradius
        dist=distance.cdist(b,b,"euclidean")
        for i in range(len(b)):
            if any(i in s for s in near):
                pass
            else:
                near.append([i])
                if i==(len(b)-1):
                    pass
                else:
                    for j in range(i+1,len(b)):
                        if vald[i]>dist[i,j]:
                            #extracheck possible here
                            near[len(near)-1].append(j)
        newval=np.zeros(len(val))                        
        for i in range(len(near)):
            for j in range(len(near[i])):
                newval[near[i][j]]=np.sum(val[near[i]])
        print near        
        maxi[:,:]=0
        for i in range(len(b)):
            maxi[b[i][0],b[i][1]]=newval[i]
        
    count=0
    nn=np.zeros([4]+form)
    nn2=np.zeros([4]+form)

    nn2[0,:,:-2]=a[:,2:]
    nn2[0,:,-2:]=-np.inf
    nn2[1,:,2:]=a[:,0:-2]
    nn2[1,:,:2]=-np.inf
    nn2[2,2:,:]=a[0:-2,:]
    nn2[2,:2,:]=-np.inf    
    nn2[3,:-2,:]=a[2:,:]
    nn2[3,-2:,:]=-np.inf
    
    nn[0,:,0:-1]=a[:,1:]
    nn[0,:,-1]=-np.inf
    nn[1,:,1:]=a[:,0:-1]
    nn[1,:,0]=-np.inf
    nn[2,1:,:]=a[0:-1,:]
    nn[2,0,:]=-np.inf    
    nn[3,0:-1,:]=a[1:,:]
    nn[3,-1,:]=-np.inf
    
    
    ueber=np.zeros([4]+form)
    for i in range(4):
        nn2[i,:,:]=(nn2[i,:,:]-a)>0
        nn[i,:,:]=(nn[i,:,:]-a)>0

    nn=(nn+nn2)>0
    neu=np.zeros([4]+form)

    
    cn=np.zeros(form)
    
    while count<lim:
        oldmaxi=np.copy(maxi)
        
        neu[0,:,0:-1]=maxi[:,1:]
        neu[1,:,1:]=maxi[:,0:-1]
        neu[2,1:,:]=maxi[0:-1,:]
        neu[3,0:-1,:]=maxi[1:,:]
                
        
        for i in range(4):
            ueber[i,:,:]=nn[i,:,:]*neu[i,:,:]        
            cn+=(ueber[i,:,:]-cn)*(ueber[i,:,:]-maxi>0)*(cn<ueber[i,:,:])###        
        
        maxi+=cn*((maxi==0)*1)
        if np.sum(np.abs(maxi-oldmaxi))==0:
            print 'break after '+str(count)
            break
        
        cn[:,:]=0
        count+=1
    return maxi,np.vstack((b.T[1],b.T[0]))


def isleft(p0,p1,ptest):
    '''groesser 0 für ptest links der Linie von p0 zu p1'''
    return (p1[0]-p0[0])*(ptest[1]-p0[1])-(ptest[0]-p0[0])*(p1[1]-p0[1])

def eratos(n):
    m=np.ones(n)
    for i in range(2,int(np.sqrt(n)+0.99)):
        if m[i]==1:
            for j in range(int((n-i**2)/i+1)):
                if i**2+j*i<len(m):
                    m[i**2+j*i]=0
    m=m*np.arange(n)
    m=m[m>1]
    return m
