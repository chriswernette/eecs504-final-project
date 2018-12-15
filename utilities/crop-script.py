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

def line_select_callback(eclick, erelease):
    global x1, y1, x2, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

path = '../data/Test_Images/_021_first_15/'
files = os.listdir(path)
files.sort()
num_files = len(files)
for i in range(len(files)):
    files[i] = path + files[i]
    print(files[i])

coords = np.array([0,0,0,0])

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
        ax, line_select_callback, drawtype='box',
        rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
    fig = plt.gcf()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    # fig.tight_layout()
    # fig.savefig(new_name,bbox_inches='tight')

    if(i == 0):
        coords[:] = x1,x2,y1,y2
        coords = np.floor(coords).astype(np.int)
    else:
        to_add = np.floor(np.array([x1,x2,y1,y2])).astype(int)
        coords = np.vstack((coords,to_add))
    plt.close(fig) 

    filename = path + 'crops.npy'
    np.save(filename, coords)