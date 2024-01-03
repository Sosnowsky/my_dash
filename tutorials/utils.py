import numpy as np
import fppanalysis as fppa
import h5py
import xarray as xr
import plotly.graph_objects as go


def vector_rotation(vector, beta):
    """
    Code from Seung-Gyou Baek from 'test_tde.py'
    """
    beta_rad = beta * (np.pi / 180)
    M = np.zeros((2, 2))
    M[0, 0] = np.cos(beta_rad)
    M[1, 0] = np.sin(beta_rad)
    M[0, 1] = -np.sin(beta_rad)
    M[1, 1] = np.cos(beta_rad)
    vector_prime = M.dot(vector)
    return vector_prime


def rad_pol_positions(filename):
    """
    Create radial and poloidal positions of the pixels.
    Code from Seung-Gyou Baek in 'test_tde.py'
    """
    z_arr = filename["z_arr"] / 10.0  # in centimeters, shape = (16,8), (row, column)
    r_arr = filename["r_arr"] / 10.0  # in centimeters, (16,8), (row,column)
    pol_arr = z_arr * 0.0
    rad_arr = r_arr * 0.0
    nrows = r_arr.shape[0]
    ncols = r_arr.shape[1]
    fov_angle = 21.485
    for j in range(ncols):
        for i in range(nrows):
            vector_prime = vector_rotation(
                [(r_arr[i, j] - r_arr[15, 0]), z_arr[i, j] - z_arr[15, 0]], fov_angle
            )
            pol_arr[i, j] = vector_prime[1]
            rad_arr[i, j] = vector_prime[0]
    return z_arr, r_arr, pol_arr, rad_arr


def create_xarray_from_hdf5(filename, R_arr, Z_arr):
    """
    Creates an xarray dataset from the W7X HDF5 files.

    Code from Seung-Gyou Baek in 'test_tde.py'
        Input:
            - filename: string, .h5 file
            - R_arr: radial positions of pixels
            - Z_arr: poloidal positions of pixels
        Returns:
            - ds: xarray dataset
    """
    f = h5py.File(filename, "r")
    frames = f["frames"][()]
    times = f["dimensions"][()]  # times are in units of nanoseconds
    time = (times - times[0]) / 1.0e9 - 0.992  # in seconds
    scale = f["scale"][()]
    offset = f["offset"][()]
    # I want to take 1001 samples to be consistent with the IDL script
    offsets = np.sum(frames[0:1001, :, :], axis=0) / 1001.0
    # make the brightness unit from int to float
    # in order to apply the scale and offset factors
    bri = frames.astype(float)
    for i in range(bri[:, 0, 0].shape[0]):  # (frames, row, column)
        bri[i, :, :] = -scale * (offsets - bri[i, :, :])
    return xr.Dataset(
        {"frames": (["time", "y", "x"], bri)},
        coords={
            "R": (["y", "x"], R_arr),
            "Z": (["y", "x"], Z_arr),
            "time": (["time"], time),
        },
    )


def run_norm_ds(ds, radius):
    """Returns running normalized dataset of a given dataset using run_norm from
    fppanalysis function by applying xarray apply_ufunc. Thanks to the
    Research software engineering (RSE) group at UiT for help with this.
    Input:
        - ds: xarray Dataset
        - radius: radius of the window used in run_norm. Window size is 2*radius+1. ... int
    'run_norm' returns a tuple of time base and the signal. Therefore, apply_ufunc will
    return a tuple of two DataArray (corresponding to time base and the signal).
    To return a format like the original dataset, we create a new dataset of normalized frames and
    corresponding time computed from apply_ufunc.
    Description of apply_ufunc arguments.
        - first the function
        - then arguments in the order expected by 'run_norm'
        - input_core_dimensions: list of lists, where the number of inner sequences must match
        the number of input arrays to the function 'run_norm'. Each inner sequence specifies along which
        dimension to align the corresponding input argument. That means, here we want to normalize
        frames along time, hence 'time'.
        - output_core_dimensions: list of lists, where the number of inner sequences must match
        the number of output arrays to the function 'run_norm'.
        - exclude_dims: dimensions allowed to change size. This must be set for some reason.
        - vectorize must be set to True in order to for run_norm to be applied on all pixels.
    """

    normalization = xr.apply_ufunc(
        fppa.run_norm,
        ds["frames"],
        radius,
        ds["time"],
        input_core_dims=[["time"], [], ["time"]],
        output_core_dims=[["time"], ["time"]],
        exclude_dims=set(("time",)),
        vectorize=True,
    )

    ds_normalized = xr.Dataset(
        data_vars=dict(
            frames=(["y", "x", "time"], normalization[0].data),
        ),
        coords=dict(
            R=(["y", "x"], ds["R"].data),
            Z=(["y", "x"], ds["Z"].data),
            time=normalization[1].data[0, 0, :],
        ),
    )

    return ds_normalized


def is_dead(x):
    return len(x) == 0 or np.isnan(x[0])


def is_within_boundaries(ds, x, y):
    return 0 <= x < ds.dims["x"] and 0 <= y < ds.dims["y"]


def add_pixels(ds, fig):
    radial_pos, pol_pos = ds.R.values, ds.Z.values
    dead_pixels = [
        is_dead(ds.sel(x=i, y=j)["frames"].values)
        for j in range(ds.dims["y"])
        for i in range(ds.dims["x"])
    ]
    alive_pixels = np.invert(dead_pixels)
    texts = np.array(
        [
            "{} {}".format(y, x)
            for x in range(0, radial_pos.shape[0])
            for y in range(0, radial_pos.shape[1])
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=radial_pos.flatten()[alive_pixels],
            y=pol_pos.flatten()[alive_pixels],
            mode="markers",
            text=texts[alive_pixels],
            marker_size=5,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=radial_pos.flatten()[dead_pixels],
            y=pol_pos.flatten()[dead_pixels],
            mode="markers",
            text=texts[dead_pixels],
            marker_size=5,
            marker_symbol="x",
        )
    )
