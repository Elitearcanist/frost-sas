import sys

import matplotlib.pyplot as plt
import numpy as np
from data_importing.read_stave import read_stave_data_hdf5


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
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

    pos1 = ax1.imshow(
        abs(data.data), cmap="BuPu", aspect="auto"
    )  # , extent=[-1,1,-10,10])
    ax1.set_title("Magnitude")
    ax1.set_xlabel("Pulses * N_Receivers")
    ax1.set_ylabel("Sample")
    fig.colorbar(pos1, ax=ax1)
    # forceAspect(ax1, aspect=1.0)
    # fig.savefig("magnitude.png", bbox_inches='tight')
    # plt.show()

    #
    # Plot phase
    #
    if data.data.dtype == np.dtype("complex128"):
        pos2 = ax2.imshow(
            np.angle(data.data), cmap="BuPu", aspect="auto"
        )  # , extent=[-1,1,-10,10])
        ax2.set_title("Phase")
        ax2.set_xlabel("Pulses * N_Receivers")
        fig.colorbar(pos2, ax=ax2)  # TODO data is in radians
        # forceAspect(ax2, aspect=1.0)
        # fig.savefig("phase.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plotHdf5()
