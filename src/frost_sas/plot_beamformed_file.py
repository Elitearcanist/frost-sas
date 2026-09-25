# Copied directly from MASTODON tools

import sys

import matplotlib.pyplot as plt
import numpy as np
from data_importing.read_beamformed import readBeamFile

if len(sys.argv) == 1:
    # print('Must specify beamformed filename on command line')
    # exit()
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename()
else:
    filename = sys.argv[1]

[x, y, data] = readBeamFile(filename)
data = np.abs(data)
data_max = np.max(data)
data = data / data_max
# data = np.clip(data, 0, 0.25)  # @todo: Remove - specific to one type of beamformed output

#
# Plot magnitude
#
fig = plt.figure()
ax = fig.add_subplot(111)

data = np.flipud(data)

# ax.imshow(np.log10(data), cmap = 'gray')#, extent=[-1,1,-10,10])
ax.imshow(
    data, extent=[x[0], x[-1], y[0], y[-1]], cmap="BuPu"
)  # , extent=[-1,1,-10,10])
print(x[-1])
plt.title("Beamformed")
# forceAspect(ax,aspect=3.5)  # @todo: Remove - specific to one type of beamformed output
# fig.savefig("magnitude.png", bbox_inches='tight')
plt.show()
