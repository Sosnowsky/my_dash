import numpy as np
import velocity_estimation as ve
import xarray as xr
import h5py
from utils import *
from dash import Dash, Input, Output, callback, dcc, html

import plotly.express as px
import pandas as pd

import plotly.figure_factory as ff
import plotly.graph_objects as go

shot = 221214013
filename = "/home/sosno/Data/221214013.h5"
rad_pol_filename = np.load("/home/sosno/Data/rz_arrs.npz")
z_arr, r_arr, pol_arr, rad_arr = rad_pol_positions(rad_pol_filename)
ds = create_xarray_from_hdf5(filename, rad_arr, pol_arr)

t_start = 7.1
t_end = 7.105
ds = ds.sel(time=slice(t_start, t_end))
ds = run_norm_ds(ds, 1000)

eo = ve.EstimationOptions()
eo.method = ve.TDEMethod.CC
eo.cc_options.minimum_cc_value = 0

md = ve.estimate_velocity_field(ve.CModImagingDataInterface(ds), eo)

vx = md.get_vx()
vy = md.get_vy()
conf = md.get_confidences()
R = md.get_R()
Z = md.get_Z()
range_r, range_z = range(0, R.shape[0]), range(0, R.shape[1])


fig = ff.create_quiver(R, Z, vx, vy,
                       scale=5e-6,
                       name='quiver',
                       line_width=1,
                       hovertemplate=f'conf: %{range_r}')




app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Input(id="scale", type="number", placeholder="", value=5e-6, style={'marginRight': '10px'}),
        dcc.Graph(id="quiver", figure=fig, style={'width': '90vh', 'height': '90vh'}),
        dcc.Input(id="x1", type="number", placeholder="x1", value=5, style={'marginRight': '10px'}),
        dcc.Input(id="y1", type="number", placeholder="y1", value=5, style={'marginRight': '10px'}),
        dcc.Input(id="x2", type="number", placeholder="x2", value=5, style={'marginRight': '10px'}),
        dcc.Input(id="y2", type="number", placeholder="y2", value=6, style={'marginRight': '10px'}),
        dcc.Graph(id="ccf", figure={}, style={'width': '90vh', 'height': '90vh'}),
    ]
)


@callback(
    Output("quiver", "figure"),
    Input("scale", "value"),
)
def update_scale(scale):
    fig = ff.create_quiver(R, Z, vx, vy,
                     scale=scale,
                     name='quiver',
                     line_width=1,
                     hovertemplate=f'conf: %{range_r}')
    fig.add_trace(go.Scatter(
        x=R,
        y=Z,
        marker_size=np.ones(shape=R.shape),
    ))
    return fig


@callback(
    Output("ccf", "figure"),
    Input("x1", "value"),
    Input("y1", "value"),
    Input("x2", "value"),
    Input("y2", "value"),
)
def update_ccf(x1, y1, x2, y2):
    s1 = ds.isel(x=x1, y=y1)["frames"].values
    s2 = ds.isel(x=x2, y=y2)["frames"].values
    ccf_times, ccf = fppa.corr_fun(s1, s2, 5e-7)
    return px.line(dict(y=ccf, t=ccf_times), x="t", y="y")


if __name__ == "__main__":
    app.run()
