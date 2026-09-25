import sys

import matplotlib.pyplot as plt
import numpy as np
from read_stave import read_stave_data_hdf5


# From https://moonbooks.org/Articles/How-to-change-imshow-aspect-ratio-in-matplotlib-/
def forceAspect(ax, aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plotHdf5():
    if len(sys.argv) == 1:
        # print('Must specify stave data .json file on command line')
        # exit()
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        filename = filedialog.askopenfilename()
    else:
        filename = sys.argv[1]

    # with open(filename, 'r') as json_file:
    #    stave = json.load(json_file)
    #
    # data = stave['data']
    # data = np.array(data)

    data = read_stave_data_hdf5(filename)

    #
    # Plot magnitude
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(abs(data.data))  # , extent=[-1,1,-10,10])
    plt.title("Magnitude")
    forceAspect(ax, aspect=1.0)
    # fig.savefig("magnitude.png", bbox_inches='tight')
    plt.show()

    #
    # Plot phase
    #
    if data.data.dtype == np.dtype("complex128"):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.imshow(np.angle(data.data))  # , extent=[-1,1,-10,10])
        plt.title("Phase")
        forceAspect(ax, aspect=1.0)
        # fig.savefig("phase.png", bbox_inches='tight')
        plt.show()
