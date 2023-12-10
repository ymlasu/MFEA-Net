import numpy as np
import matplotlib.pyplot as plt

def plot_elasticfield(img,ftitle='Title',f1title='Field-1',f2title='Field-2',fname=None,limit1=None,limit2=None):
    '''Plot the 2D pytorch tensor field (assume there are two fields, with shape of 2 x n x n)'''

    img = img.cpu()
    fig = plt.figure()
    fig.suptitle(ftitle)

    fig.add_subplot(1,2,1)
    if(limit1 is None):
        im1 = plt.imshow(img[0])
    else:
        im1 = plt.imshow(img[0], vmin=limit1[0],vmax=limit1[1])
    plt.axis('off')
    plt.title(f1title)
    plt.colorbar(im1, fraction=0.045)

    fig.add_subplot(1,2,2)
    if(limit2 is None):
        im2 = plt.imshow(img[1])
    else:
        im2 = plt.imshow(img[1],vmin=limit2[0],vmax=limit2[1])
    plt.axis('off')
    plt.title(f2title)
    plt.colorbar(im2, fraction=0.045)

    plt.tight_layout()

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight',dpi=500)
    plt.show()

def plot_thermalfield(field, ftitle='Title'):
    '''Default is to plot the solution field'''

    field_size = field.size()
    if(len(field_size) == 3):
        field = field.reshape(field_size[1:])
        
    field = field.cpu()
    fig = plt.figure()
    fig.suptitle(ftitle)

    im = plt.imshow(field)
        
    plt.axis('off')
    plt.colorbar(im)
    plt.show()


def heatmap2d(arr, fname, cbar = True):
    plt.imshow(arr,cmap='viridis',vmin=-1,vmax=1)
    if(cbar is True):
        plt.colorbar()
    plt.gcf()
    plt.axis('off')
    plt.savefig(fname, dpi=500, bbox_inches='tight')
    plt.show()