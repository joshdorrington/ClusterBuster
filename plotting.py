import cartopy.crs as ccrs
import iris.plot as iplt
import cmocean.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def quick_contourf(cube,cmap=cm.balance,clevs=None,ax=None):
    
    proj=ccrs.NearsidePerspective(central_latitude=60,central_longitude=-20)
    
    if ax is None:
        fig=plt.figure()
        ax=plt.subplot(1,1,1,projection=proj)
        
    if clevs is None:
        clevs=np.linspace(-abs(cube.data).max(),abs(cube.data).max(),21)
        
    ax.coastlines()
    ax.set_global()

    plot=iplt.contourf(cube, levels=clevs, cmap=cmap,extend="both",axes=ax)
    cbar=plt.colorbar(orientation="horizontal",mappable=plot,ax=ax)
    return plot,ax

def quick_reg_plot(cube,figdims=(15,10)):
    
    proj=ccrs.NearsidePerspective(central_latitude=60,central_longitude=-20)

    K=np.shape(cube)[0]
    fig=plt.figure()
    axes=[]
    for i in range(K):
        ax=plt.subplot(1,K,i+1,projection=proj)
        quick_contourf(cube[i],ax=ax)
        ax.set_title(f"Cluster {i+1}")
        axes.append(ax)
    fig.set_figwidth(figdims[0])
    fig.set_figheight(figdims[1])
    return fig,axes