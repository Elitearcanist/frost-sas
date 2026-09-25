# Copied directly from MASTODON tools

import json
import os

import h5py
import numpy as np


def readBeamFile(filename):
    ext = os.path.splitext(filename)[1]
    if ext.lower() == ".json":
        return readBeamJson(filename)
    if ext.lower() == ".h5" or ext.lower() == ".hdf5":
        return readBeamHdf5(filename)
    # Assume text
    return readBeamText(filename)


def readBeamText(filename):
    with open(filename).readlines() as filelines:
        for i in range(len(filelines)):
            data = filelines[i]
            data = data.replace("i", "j")
            data = data.split()
            data = np.array([np.complex(c) for c in data])

            if i == 0:
                result = data
            else:
                result = np.vstack((result, data))
        data = result

        x = np.array(range(data.shape[1]))
        y = np.array(range(data.shape[0]))

        return [x, y, data]


def readBeamJson(filename):
    with open(filename) as f:
        dataFile = json.load(f)
        x = np.array(dataFile["x_coords"])
        y = np.array(dataFile["y_coords"])
        data = np.array(dataFile["data"])

        re = data[::, 0::2]
        im = data[::, 1::2]
        data = re + 1.0j * im

        return [x, y, data]


def readBeamHdf5(filename):
    f = h5py.File(filename, "r")
    x = np.array(f["/Sonar/Sensor1/Data1/x_coords"])
    y = np.array(f["/Sonar/Sensor1/Data1/y_coords"])
    data = f["/Sonar/Sensor1/Data1/PingData"]
    data = np.array(data)
    print(data.shape)
    print(x.size)
    print(y.size)
    data = np.reshape(data, (x.size, y.size))
    data = np.transpose(data)

    return [x, y, data]
