import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.widgets as widgets
import os

def onselect(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    ax.set_ylim(erelease.ydata,eclick.ydata)
    ax.set_xlim(eclick.xdata,erelease.xdata)
    fig.canvas.draw()

path = '../data/_027_12_04_20_to_12_04_30/'
files = os.listdir(path)
files.sort()
num_files = len(files)
for i in range(len(files)):
    files[i] = path + files[i]
    print(files[i])

#loop through the selected files
for i in range(num_files):
    img_location = files[i]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename_w_ext = os.path.basename(img_location)
    filename, file_extension = os.path.splitext(filename_w_ext)
    new_name = 'cropped-dataset/' + filename + '_croppped' + '.jpg'

    im = Image.open(img_location)
    arr = np.asarray(im)
    plt_image=plt.imshow(arr)
    rs=widgets.RectangleSelector(
        ax, onselect, drawtype='box',
        rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
    fig = plt.gcf()
    plt.show()
    fig.savefig(new_name)

    plt.close(fig) 