import json

import numpy as np
import velocity_estimation as ve
import xarray as xr
import h5py
from utils import *
from dash import Dash, Input, Output, callback, dcc, html

import plotly.express as px
from plotly.subplots import make_subplots
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


fig = ff.create_quiver(R, Z, vx, vy,
                       scale=5e-6,
                       name='quiver',
                       line_width=1,
                       hovertemplate="")

fig.add_trace(go.Scatter(
    x=R.flatten(),
    y=Z.flatten(),
    mode="markers",
    text=["{} {}".format(y, x) for x in range(0, R.shape[0]) for y in range(0, R.shape[1])],
    marker_size=5,
))

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Input(id="scale", type="number", placeholder="", value=5e-6, style={'marginRight': '10px'}),
        html.Div([
            html.Div([dcc.Graph(id="quiver", figure=fig)], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id="ccf", figure={})], style={'width': '49%', 'display': 'inline-block'})])
    ]
)


@callback(
    Output("quiver", "figure"),
    Input("scale", "value"),
)
def update_scale(scale):
    fig_update = ff.create_quiver(R, Z, vx, vy,
                     scale=scale,
                     name='quiver',
                     line_width=1,
                     hovertemplate="")

    fig_update.add_trace(go.Scatter(
        x=R.flatten(),
        y=Z.flatten(),
        mode="markers",
        text=["{} {}".format(y, x) for x in range(0, R.shape[0]) for y in range(0, R.shape[1])],
        marker_size=5,
    ))

    return fig_update


@callback(
    Output("ccf", "figure"),
    Input('quiver', 'clickData'))
def display_hover_data(hd):
    i, j = [int(s) for s in hd['points'][0]['text'].split(' ')]
    print("CCF for pixel {} {}".format(i, j))

    def get_trace_for_points(x1, y1, x2, y2):
        s1 = ds.isel(x=x1, y=y1)["frames"].values
        s2 = ds.isel(x=x2, y=y2)["frames"].values
        ccf_times, ccf = fppa.corr_fun(s1, s2, 5e-7)
        return go.Scatter(x=ccf_times, y=ccf, mode="lines")
    ccf_fig = make_subplots(rows=2, cols=2)
    ccf_fig.add_trace(get_trace_for_points(i, j, i-1, j), row=1, col=1)
    ccf_fig.add_trace(get_trace_for_points(i, j, i+1, j), row=2, col=1)
    ccf_fig.update_xaxes(range=[-5e-6, 5e-6])
    return ccf_fig


if __name__ == "__main__":
    app.run()
