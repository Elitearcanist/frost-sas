import logging
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def read_stave_data_hdf5(filename: Path) -> SimpleNamespace:
    """_summary_

    Args:
        filename (Path): _description_

    Raises:
        KeyError: _description_
        ValueError: _description_
        ValueError: _description_
        e: _description_

    Returns:
        SimpleNamespace: _description_
    """

    # Test whether this is actually a beamformed data file. It's easy to
    # get the two mixed up.
    with h5py.File(filename, "r") as f:
        b_test_bf = False
        try:
            # h5read(filename, "/Sonar/Sensor1/Data1/x_coords")
            print(f["/Sonar/Sensor1/Data1/x_coords"][()])
            b_test_bf = True
        except KeyError:
            logger.info("The input file appears to ba a stave file.")

        if b_test_bf:
            raise KeyError(
                "The input file appears to be a beamformed data file, not a stave data file."
            )

        stave = SimpleNamespace()

        stave.n_time = float(f["/signal/num_samples"][0])
        stave.n_elem = float(f["/sonars/receiver/nelem"][0])

        try:
            stave.n_elem_hor = float(f["/sonars/receiver/nelem_hor"][0])
        except KeyError:
            stave.n_elem_hor = 1.0

        try:
            stave.n_elem_ver = float(f["/sonars/receiver/nelem_ver"][0])
        except KeyError:
            stave.n_elem_ver = 1.0

        stave.n_pings = float(f["/vehicle/num_pings"][0])
        stave.fc = f["/sonars/projector/frequency"][()]
        stave.bw = f["/sonars/projector/bandwidth"][()]
        stave.fs = f["/signal/fs"][()]
        stave.decimation = f["/signal/decimation"][()]
        stave.dy = f["/vehicle/dping"][()]
        stave.r_min = f["/signal/rangemin"][()]
        stave.c = f["/environment/soundspeed"][()]
        stave.signal_applied = f["/signal/signal_applied"][()]
        stave.b_basebanded = f["/signal/basebanded"][()]
        stave.proj_width = f["/sonars/projector/width"][()]
        stave.recv_width = f["/sonars/receiver/width"][()]
        # These two are deprecated, and stave.proj.* should be used instead
        stave.proj_height = f["/sonars/projector/height"][()]
        stave.recv_height = f["/sonars/receiver/height"][()]

        # These don't exist in older HDF5 files
        stave.proj = SimpleNamespace()
        stave.recv = SimpleNamespace()
        try:
            stave.proj.height = stave.proj_height
            stave.recv.height = stave.recv_height
            stave.proj.bearing = f["/sonars/projector/bearing"][()]
            stave.recv.bearing = f["/sonars/receiver/bearing"][()]
            stave.proj.depression = f["/sonars/projector/depression"][()]
            stave.recv.depression = f["/sonars/receiver/depression"][()]

            stave.veh_altitude = f["/vehicle/altitude"][()]
            stave.water_depth = f["/environment/waterdepth"][()]
        except KeyError:
            pass

        try:
            stave.proj.positions = f["/sonars/projector/positions"][()]
            stave.recv.positions = f["/sonars/receiver/positions"][()]
        except KeyError:
            # @todo: If positions are not specified, we should provide
            # positions here based on the parameters above.
            pass

        try:
            stave.hor_spacing = float(f["/sonars/receiver/hor_spacing"][0])
        except KeyError:
            stave.hor_spacing = stave.recv_width

        # Added 2022-AUG-31. Gives the positions of each element at every ping or navigation sample
        try:
            srcpos = f["/vehicle/source_positions"][()]
            stave.proj.pos = SimpleNamespace()
            stave.proj.pos.world = srcpos[:, 1:]
            stave.proj.pos.time = srcpos[:, 0]
            rcvpos = f["/vehicle/receiver_positions"][()]
            stave.recv.pos = SimpleNamespace()
            stave.recv.pos.world = rcvpos[:, 1:]
            stave.recv.pos.time = rcvpos[:, 0]
        except KeyError:
            pass

        # This is an optional field
        try:
            stave.overlap = float(f["/vehicle/sas_overlap"][()])
        except KeyError:
            pass

        if stave.signal_applied:
            stave.signal = SimpleNamespace()
            stave.signal.times = f["/signal/times"][()]
            stave.signal.signal = f["/signal/signal"][()]
            stave.signal.pulse_length = f["/signal/pulse_length"][()]
            sig_data = stave.signal.signal
            # Matlab complex(re, im) equivalent
            stave.signal.signal = sig_data["re"] + 1j * sig_data["im"]

        # Data is separate real/imaginary doubles - convert to complex
        stave.data = f["/data"][()]

        try:
            stave.signal.data_format = f["/stave/signal/data_format"][()]
            # Handle byte strings from h5py
            if isinstance(stave.signal.data_format, bytes):
                stave.signal.data_format = stave.signal.data_format.decode("utf-8")

            if stave.signal.data_format == "real":
                # Data only has a real part - use as is?
                raise ValueError("Real data in HDF5 has not be tested")
            elif stave.signal.data_format == "complex":
                # Data is interleaved real/imaginary doubles - convert to complex
                stave.data = stave.data["re"] + 1j * stave.data["im"]
            else:
                raise ValueError("Invalid data format key")
        except Exception as e:
            if str(e) in [
                "Real data in HDF5 has not be tested",
                "Invalid data format key",
            ]:
                raise e
            # Older formats do not have the data_format dataset, so assume complex.
            if not hasattr(stave, "signal"):
                stave.signal = SimpleNamespace()
            stave.signal.data_format = "complex"
            # Check if data has re/im fields before converting
            if hasattr(stave.data, "dtype") and "re" in stave.data.dtype.names:
                stave.data = stave.data["re"] + 1j * stave.data["im"]

        motion = f["/vehicle/motion"][()]
        m = []
        for n in range(len(motion["time"])):
            item = SimpleNamespace()
            item.time = motion["time"][n]
            item.x = motion["x"][n]
            item.y = motion["y"][n]
            item.z = motion["z"][n]
            item.roll = motion["roll"][n]
            item.pitch = motion["pitch"][n]
            item.yaw = motion["yaw"][n]
            item.speed = motion["speed"][n]
            m.append(item)
        stave.motion = np.array(m)
        stave.motion2 = motion

        motion_world = f["/vehicle/motion_world"][()]
        mw = []
        for n in range(len(motion_world["time"])):
            item = SimpleNamespace()
            item.time = motion_world["time"][n]
            item.latitude = motion_world["latitude"][n]
            item.longitude = motion_world["longitude"][n]
            item.depth = motion_world["depth"][n]
            item.altitude = motion_world["altitude"][n]
            item.roll = motion_world["roll"][n]
            item.pitch = motion_world["pitch"][n]
            item.heading = motion_world["heading"][n]
            item.speed = motion_world["speed"][n]
            mw.append(item)
        stave.motion_world = np.array(mw)
        stave.motion_world2 = motion_world

        # This doesn't exist in older HDF5 files
        try:
            stave.navigation = f["/vehicle/navigation"][()]
        except KeyError:
            pass

    return stave


def main():

    dataPath = Path.cwd() / "mastodon_config/StaveData.h5"

    read_stave_data_hdf5(dataPath)


main()
