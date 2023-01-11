### Import Packages ###
import numpy as n
from matplotlib import pyplot as p
from matplotlib import cm
import lensdemo_funcs as ldf
from matplotlib.widgets import Slider, Button
import cv2

### Images Preferences ###
#myargs = {'interpolation': 'nearest', 'origin': 'lower', 'cmap': cm.nipy_spectral} # Rainbow version
myargs = {'interpolation': 'nearest', 'origin': 'lower', 'cmap': cm.gray} #B/W version

### Images Coordinates ###
nx = 1000 #X resolution
ny = 1000 #Y resolution
xhilo = [-2.5, 2.5]
yhilo = [-2.5, 2.5]
x = (xhilo[1] - xhilo[0]) * n.outer(n.ones(ny), n.arange(nx)) / float(nx-1) + xhilo[0]
y = (yhilo[1] - yhilo[0]) * n.outer(n.arange(ny), n.ones(nx)) / float(ny-1) + yhilo[0]

### Set some Gaussian profile parameters for our Source ###
S_amp = 1.   # peak brightness 

S_sig = 0.1 # Gaussian "sigma" (this is more of a Source Size parameter)
S_xcen = 0.0  # X coordinate for the Source Center 
S_ycen = 0.0  # Y coordinate for the Source Center
S_ratio = 1. # Axis Ratio
S_pa = 0.0    # Counter-Clockwise Position Angle of the Source's major axis
Spar = n.asarray([S_amp, S_sig, S_xcen, S_ycen, S_ratio, S_pa]) #Creates an array of parameters
S_image = ldf.gauss_2d(x, y, Spar) #Creates an image for the Source

### Set some SIE-model parameters for our lens ###
L_amp = 1.5  # Einstein radius
L_xcen = 0.0  # X coordinate for the Lens Center
L_ycen = 0.0  # Y coordinate for the Lens Center
L_ratio = 1. # Axis Ratio 
L_pa = 0.0    # Counter-Clockwise Position Angle of the Lens' major axis
Lpar = n.asarray([L_amp, L_xcen, L_ycen, L_ratio, L_pa])

### Compute the lensing potential gradients and create the lensed image ###
(xg, yg) = ldf.sie_grad(x, y, Lpar) #Returns a tuple with gradients at positions (x,y)
L_image = ldf.gauss_2d(x-xg, y-yg, Spar) #Creates the lensed image


### Draw the images and the Sliders with relative axes ###
f = p.figure()
f.set_size_inches(10., 7.5)
p.subplots_adjust(bottom=0.45)

ax1 = f.add_subplot(1,2,1)
ax1.imshow(S_image, **myargs)
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
p.text(0.5, 1.05, 'Source Image',fontsize=12, horizontalalignment='center', transform=ax1.transAxes)


ax2 = f.add_subplot(1,2,2)
ax2.imshow(L_image, **myargs)
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
p.text(0.5, 1.05, 'Lensed Image',fontsize=12, horizontalalignment='center', transform=ax2.transAxes)

axcolor = 'lightgoldenrodyellow'

### Source Axes ###
axSxcen = p.axes([0.175, 0.3, 0.25, 0.03], facecolor=axcolor)
axSycen = p.axes([0.175, 0.25, 0.25, 0.03], facecolor=axcolor)
axSsigma = p.axes([0.175, 0.2, 0.25, 0.03], facecolor=axcolor)
axSratio = p.axes([0.175, 0.15, 0.25, 0.03], facecolor=axcolor)
axSpa = p.axes([0.175, 0.1, 0.25, 0.03], facecolor=axcolor)
p.text(0.5, 1.75, 'Source Parameters', horizontalalignment='center', transform=axSxcen.transAxes)

### Lens Axes ###
axLxcen = p.axes([0.6, 0.3, 0.25, 0.03], facecolor=axcolor)
axLycen = p.axes([0.6, 0.25, 0.25, 0.03], facecolor=axcolor)
axLamp = p.axes([0.6, 0.2, 0.25, 0.03], facecolor=axcolor)
axLratio = p.axes([0.6, 0.15, 0.25, 0.03], facecolor=axcolor)
axLpa = p.axes([0.6, 0.1, 0.25, 0.03], facecolor=axcolor)
p.text(0.5, 1.75, 'Lens Parameters', horizontalalignment='center', transform=axLxcen.transAxes)

### Source Slider System ###
S_xcen_U = Slider(axSxcen, 'Source X', -2., 2., valinit=0., valstep=0.1)
S_ycen_U = Slider(axSycen, 'Source Y', -2., 2., valinit=0., valstep=0.1)
S_sigma_U = Slider(axSsigma, 'Source Intensity', 0., 0.2, valinit=0.1, valstep=0.05)
S_ratio_U = Slider(axSratio, 'Axes Ratio', 0.1, 10., valinit=1., valstep=0.1)
S_pa_U = Slider(axSpa, 'Position Angle', 0., 360., valinit=0., valstep=1.)

### Lens Slider System ###
L_xcen_U = Slider(axLxcen, 'Lens X', -2., 2., valinit=0., valstep=0.1)
L_ycen_U = Slider(axLycen, 'Lens Y', -2., 2., valinit=0., valstep=0.1)
L_amp_U = Slider(axLamp, 'Lens Strength', 0., 3., valinit=1.5, valstep=0.1)
L_ratio_U = Slider(axLratio, 'Axes Ratio', 0.1, 10., valinit=1., valstep=0.1)
L_pa_U = Slider(axLpa, 'Position Angle', 0., 360., valinit=0, valstep=1.)

### Reset Button ###
resetax = p.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


### Update Function ###
def update(val):
        
    ### Set some UPDATED Gaussian profile parameters for our Source ###
    Spar = n.asarray([S_amp, S_sigma_U.val, S_xcen_U.val, S_ycen_U.val, S_ratio_U.val, S_pa_U.val])
    S_image = ldf.gauss_2d(x, y, Spar)
    
    ### Set some UPDATED SIE-model parameters for our lens ###
    Lpar = n.asarray([L_amp_U.val, L_xcen_U.val, L_ycen_U.val, L_ratio_U.val, L_pa_U.val])
    
    ### Recompute the lensing potential gradients and create the lensed image ###
    (xg, yg) = ldf.sie_grad(x, y, Lpar)
    L_image = ldf.gauss_2d(x-xg, y-yg, Spar)
    
    ### Redraw the image without opening a new window ###
    ax1.imshow(S_image, **myargs)
    ax2.imshow(L_image, **myargs)
    f.canvas.draw()
    

### Reset Function ###
def reset(event):
    S_xcen_U.reset()
    S_ycen_U.reset()
    S_sigma_U.reset()
    S_ratio_U.reset()
    S_pa_U.reset()
    
    L_xcen_U.reset()
    L_ycen_U.reset()
    L_amp_U.reset()
    L_ratio_U.reset()
    L_pa_U.reset()
    

### On_click events ###
button.on_clicked(reset)    
    
S_sigma_U.on_changed(update)
S_xcen_U.on_changed(update)
S_ycen_U.on_changed(update)
S_ratio_U.on_changed(update)
S_pa_U.on_changed(update)

L_amp_U.on_changed(update)
L_xcen_U.on_changed(update)
L_ycen_U.on_changed(update)
L_ratio_U.on_changed(update)
L_pa_U.on_changed(update)











