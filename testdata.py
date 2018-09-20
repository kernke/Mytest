# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mountain as m

#%%
x=np.linspace(-50,50,500)
y=np.linspace(-50,50,500)
xg,yg=np.meshgrid(x,y)
z=np.cos(xg*yg)+xg
z1=4*np.exp(-(xg**2+(5+yg)**2)/100)
z2=7*np.exp(-((xg-20)**2+(5+yg)**2)/50)
z3=5*np.exp(-((xg-20)**2+(25+yg)**2)/250)
z4=5*np.exp(-((xg+45)**2+(-25+yg)**2)/450)
z5=4*np.exp(-((xg+25)**2+(-35+yg)**2)/20)

z=z1+z2+z3+z4+z5+np.random.normal(0,0.01,[500,500])
#plt.imshow(z)
#%%
a=mountfast(z)

#%%
a2=mountfast(z,lim=400)
#%%
#b,c=
b,c=flood(z,a,floodthreshold=100,clustercheckradius=0.25)
plt.imshow(b)
#%%

plt.contourf(np.log(b))
plt.scatter(c[0],c[1],c='r')#