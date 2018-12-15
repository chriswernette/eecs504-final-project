import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.widgets as widgets

def onselect(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    ax.set_ylim(erelease.ydata,eclick.ydata)
    ax.set_xlim(eclick.xdata,erelease.xdata)
    fig.canvas.draw()

fig = plt.figure()
ax = fig.add_subplot(111)
name = "frame14"
filename= name + ".jpg"
new_name = name + '_croppped' + '.jpg'
plt.savefig(new_name)

im = Image.open(filename)
arr = np.asarray(im)
plt_image=plt.imshow(arr)
rs=widgets.RectangleSelector(
    ax, onselect, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
fig = plt.gcf()
plt.show()
fig.savefig(new_name)

plt.close(fig) 