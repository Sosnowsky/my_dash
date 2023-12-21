import json

import numpy as np
import velocity_estimation as ve
import xarray as xr
import h5py
from utils import *
from dash import Dash, Input, Output, callback, dcc, html, State

import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from plotly_resampler import FigureResampler

import plotly.figure_factory as ff
import plotly.graph_objects as go

shot = 221214013
filename = "/home/sosno/Data/221214013.h5"
rad_pol_filename = np.load("/home/sosno/Data/rz_arrs.npz")
z_arr, r_arr, pol_arr, rad_arr = rad_pol_positions(rad_pol_filename)
ds = create_xarray_from_hdf5(filename, rad_arr, pol_arr)
ds = ds.sel(time=slice(5, 20))
ds = run_norm_ds(ds, 1000)


def get_velocity_field(data, t_min, t_max):
    ds_small = data.sel(time=slice(t_min, t_max))

    eo = ve.EstimationOptions()
    eo.method = ve.TDEMethod.CC
    eo.cc_options.minimum_cc_value = 0.5
    eo.cc_options.running_mean = False

    md = ve.estimate_velocity_field(ve.CModImagingDataInterface(ds_small), eo)

    return md.get_R(), md.get_Z(), md.get_vx(), md.get_vy()


R, Z, vx, vy = get_velocity_field(ds, 7.1, 7.105)


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

full_times = ds["time"].values
fig_times = go.Figure(data=[go.Scatter(x=[0, 1], y=[min(full_times), max(full_times)])])
fig_times.update_yaxes(range=[min(full_times), max(full_times)])
fig_times.update_layout(autosize=False)

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id="times", figure=fig_times),
        html.Div([
            html.Button('Sync', id='sync', n_clicks=0),
        ]),
        dcc.Input(id="scale", type="number", placeholder="", value=5e-6, style={'marginRight': '10px'}),
        html.Div([
            dcc.Graph(id="quiver", figure=fig, style={'width': '49vw', 'height': '49vw', 'display': 'inline-block'}),
            dcc.Graph(id="ccf", figure={}, style={'width': '49vw', 'height': '49vw', 'display': 'inline-block'})])
    ]
)


@callback(
    Output("quiver", "figure", allow_duplicate=True),
    Input("sync", "n_clicks"),
    State('times', 'figure'),
    prevent_initial_call=True
)
def update_output(n_clicks, figure):
    t_min, t_max = figure["layout"]["yaxis"]["range"]

    print("lol")
    return update_scale(5e-6)


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


ccf_fig = FigureResampler(make_subplots(rows=2, cols=1))


@callback(
    Output("ccf", "figure"),
    Input('quiver', 'clickData'),
    State('times', 'figure'),
    prevent_initial_call=True
)
def display_hover_data(hd, figure):
    global ccf_fig
    i, j = [int(s) for s in hd['points'][0]['text'].split(' ')]
    t_min, t_max = figure["layout"]["yaxis"]["range"]
    print("CCF for pixel {} {}".format(i, j))

    def get_trace_for_points(x1, y1, x2, y2):
        s1 = ds.sel(x=x1, y=y1, time=slice(t_min, t_max))["frames"].values
        s2 = ds.sel(x=x2, y=y2, time=slice(t_min, t_max))["frames"].values
        ccf_times, ccf = fppa.corr_fun(s1, s2, 5e-7)
        return go.Scatter(x=ccf_times, y=ccf, mode="lines", name="{} {}".format(x2, y2))
    #subplot_titles = ["", "{} {}".format(i, j+1), "",
    #                  "{} {}".format(i-1, j), "", "{} {}".format(i+1, j),
    #                  "", "{} {}".format(i, j-1), ""]

    x1, y1, x2, y2 = i, j, i-1, j
    s1 = ds.sel(x=x1, y=y1, time=slice(t_min, t_max))["frames"].values
    s2 = ds.sel(x=x2, y=y2, time=slice(t_min, t_max))["frames"].values
    ccf_times, ccf = fppa.corr_fun(s1, s2, 5e-7)

    ccf_fig.add_trace(go.Scattergl(name="{} {}".format(x2, y2)), hf_x=ccf_times, hf_y=ccf, row=1, col=1)
    #ccf_fig.add_trace(get_trace_for_points(i, j, i-1, j), row=1, col=1)
    ccf_fig.add_trace(get_trace_for_points(i, j, i+1, j), row=1, col=1)
    ccf_fig.add_trace(get_trace_for_points(i, j, i, j+1), row=1, col=1)
    ccf_fig.add_trace(get_trace_for_points(i, j, i, j-1), row=1, col=1)
    ccf_fig.add_trace(get_trace_for_points(i, j, i, j), row=2, col=1)
    # ccf_fig.update_xaxes(range=[-5e-3, 5e-3])
    return ccf_fig


if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
