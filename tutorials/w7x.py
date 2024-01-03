from scipy import signal as ss
import velocity_estimation as ve
from utils import *
from dash import Dash, Input, Output, callback, dcc, html, State

from plotly_resampler import FigureResampler
from trace_updater import TraceUpdater

import plotly.figure_factory as ff
import plotly.graph_objects as go

shots = {"w7x": "221214013.h5", "09": "1160616009.nc", "16": "1160616016.nc"}
root_path = "/home/sosno/Data/"
shot = "16"
dt = 5e-7


def open_dataset(shot_number):
    if shot_number == "w7x":
        rad_pol_filename = np.load(root_path + "rz_arrs.npz")
        z_arr, r_arr, pol_arr, rad_arr = rad_pol_positions(rad_pol_filename)
        return create_xarray_from_hdf5(root_path + shots[shot_number], rad_arr, pol_arr)
    return xr.open_dataset(root_path + shots[shot_number])


ds = open_dataset(shot)


def get_velocity_field(data, _t_min, _t_max):
    ds_small = data.sel(time=slice(_t_min, _t_max))

    eo = ve.EstimationOptions()
    eo.method = ve.TDEMethod.CC
    eo.cc_options.minimum_cc_value = 0.5
    eo.cc_options.running_mean = False

    md = ve.estimate_velocity_field(ve.CModImagingDataInterface(ds_small), eo)

    return md.get_R(), md.get_Z(), md.get_vx(), md.get_vy()


R, Z = ds.R.values, ds.Z.values

fig = ff.create_quiver(
    R,
    Z,
    np.zeros(R.shape),
    np.zeros(R.shape),
    scale=5e-6,
    name="quiver",
    line_width=1,
    hovertemplate="",
)
add_pixels(ds, fig)


fig_raw = FigureResampler()
fig_raw.update_layout(autosize=False)

app = Dash(__name__)
square_style = {"width": "49vw", "height": "49vw", "display": "inline-block"}
button_style = {"width": "5vw", "padding": 10}

col_indxs = range(ds.dims["x"])
column_row = "Column"
app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown("""**Shot**"""),
                dcc.Dropdown(list(shots.keys()), shot, id="shot", style=button_style),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    ["Column", "Row"], column_row, id="column_row", style=button_style
                ),
                dcc.Dropdown(
                    np.arange(0, ds.dims["x"]), 4, id="column_indx", style=button_style
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
        html.Div(
            [
                dcc.Graph(id="raw", figure=fig_raw, style={"width": "90vw"}),
                html.Div(
                    [
                        dcc.Input(
                            id="norm_width",
                            type="number",
                            placeholder="norm width",
                            value=1000,
                            style=button_style,
                        ),
                        html.Button(
                            "Apply normalization",
                            id="apply_norm",
                            n_clicks=0,
                            style=button_style,
                        ),
                        html.Button(
                            "Reset dataset", id="reset", n_clicks=0, style=button_style
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column"},
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
        TraceUpdater(id="trace-updater-raw", gdID="raw"),
        html.Div(
            [
                html.Button("Sync", id="sync", n_clicks=0, style=button_style),
                dcc.Input(
                    id="scale",
                    type="number",
                    placeholder="scale",
                    value=5e-6,
                    style=button_style,
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
        html.Div(
            [
                dcc.Graph(id="quiver", figure=fig, style=square_style),
                dcc.Graph(id="ccf", figure={}, style=square_style),
                TraceUpdater(id="trace-updater", gdID="ccf"),
            ]
        ),
        html.Div(
            [
                dcc.Markdown("""**Selection Data**"""),
                html.Pre(id="selected_data"),
            ],
            className="three columns",
        ),
        dcc.Dropdown(["none", "PDF", "PSD"], "none", id="others_plot"),
        dcc.Graph(id="others", figure={}, style=square_style),
        html.Div(id="null", style={"display": "none"}),
    ]
)


@callback(
    Output("raw", "figure", allow_duplicate=True),
    Input("shot", "value"),
    State("column_indx", "value"),
    prevent_initial_call=True,
)
def update_shot(value, col):
    global ds
    ds = open_dataset(value)
    return update_output(col)


@callback(
    Output("raw", "figure", allow_duplicate=True),
    Input("apply_norm", "n_clicks"),
    State("norm_width", "value"),
    State("column_indx", "value"),
    prevent_initial_call=True,
)
def update_ds(n_clicks, width, col):
    global ds
    ds = run_norm_ds(ds, width)
    return update_output(col)


@callback(
    Output("raw", "figure", allow_duplicate=True),
    Input("reset", "n_clicks"),
    State("column_indx", "value"),
    State("shot", "value"),
    prevent_initial_call=True,
)
def reset_ds(n_clicks, col, shot_nm):
    global ds
    ds = open_dataset(shots[shot_nm])
    return update_output(col)


@callback(Output("null", "n_clicks"), Input("column_row", "value"))
def update_column_row(value):
    global column_row
    column_row = value
    return None


@callback(
    Output("raw", "figure"),
    Input("column_indx", "value"),
)
def update_output(col):
    plot_col = column_row == "Column"
    fig_raw.data = []
    fig_raw.update_layout(title="{} {}".format(column_row, col))
    std = 0
    for i in range(ds.dims["y" if plot_col else "x"]):
        signal = (
            ds.sel(x=col, y=i)["frames"].values
            if plot_col
            else ds.sel(x=i, y=col)["frames"].values
        )
        if not is_dead(signal):
            fig_raw.add_trace(go.Scatter(x=ds["time"].values, y=std + signal))
            std += signal.std() * 5
    return fig_raw


@callback(
    Output("quiver", "figure", allow_duplicate=True),
    Input("scale", "value"),
    Input("sync", "n_clicks"),
    State("raw", "figure"),
    prevent_initial_call=True,
)
def update_scale(scale, n_clicks, figure):
    t_min, t_max = figure["layout"]["xaxis"]["range"]
    print(
        "Computing velocity field between times {:.2f} and {:.2f}".format(t_min, t_max)
    )
    R, Z, vx, vy = get_velocity_field(ds, t_min, t_max)

    fig_update = ff.create_quiver(
        R, Z, vx, vy, scale=scale, name="quiver", line_width=1, hovertemplate=""
    )
    add_pixels(ds, fig_update)
    return fig_update


def get_indexes(text):
    return


@callback(
    Output("selected_data", "children"),
    Input("quiver", "selectedData"),
    prevent_initial_call=True,
)
def pixel_selection(data):
    if data is None:
        return []

    texts = [p["text"] for p in data["points"]]
    indexes = np.array(list(map(lambda t: [int(s) for s in t.split(" ")], texts)))
    print("indexes are {}".format(indexes))
    return " ".join(map(lambda t: "[ " + t + " ]", texts))
    # return json.dumps(indexes, indent=2)


ccf_fig = FigureResampler()


@callback(
    Output("ccf", "figure"),
    Input("quiver", "clickData"),
    State("raw", "figure"),
    prevent_initial_call=True,
)
def display_hover_data(hd, figure):
    global ccf_fig
    ccf_fig.data = []
    i, j = [int(s) for s in hd["points"][0]["text"].split(" ")]
    t_min, t_max = figure["layout"]["xaxis"]["range"]
    print("CCF for pixel {} {}, at times {:.2f} {:.2f}".format(i, j, t_min, t_max))

    def plot_trace(x1, y1, x2, y2, name):
        if not is_within_boundaries(ds, x2, y2):
            return
        # name = "{} {}".format(x2, y2)
        s1 = ds.sel(x=x1, y=y1, time=slice(t_min, t_max))["frames"].values
        s2 = ds.sel(x=x2, y=y2, time=slice(t_min, t_max))["frames"].values
        ccf_times, ccf = fppa.corr_fun(s1, s2, 5e-7)
        is_acf = (x1, y1) == (x2, y2)
        if is_acf:
            name = "acf"

        ccf_fig.add_trace(
            go.Scattergl(name=name, visible="legendonly"), hf_x=ccf_times, hf_y=ccf
        )

    plot_trace(i, j, i - 1, j, "left")
    plot_trace(i, j, i + 1, j, "right")
    plot_trace(i, j, i, j - 1, "down")
    plot_trace(i, j, i, j + 1, "up")
    plot_trace(i, j, i, j, "self")
    return ccf_fig


@callback(
    Output("others", "figure"),
    Input("others_plot", "value"),
    State("quiver", "selectedData"),
    State("raw", "figure"),
    prevent_initial_call=True,
)
def plot_others(plot, sd, raw_fig):
    t_min, t_max = raw_fig["layout"]["xaxis"]["range"]
    texts = [p["text"] for p in sd["points"]]
    indexes = np.array(list(map(lambda t: [int(s) for s in t.split(" ")], texts)))
    new_fig = go.Figure()

    if plot == "PDF":
        new_fig.update_yaxes(type="log")
        for pixel in indexes:
            signal = ds.sel(x=pixel[0], y=pixel[1], time=slice(t_min, t_max))[
                "frames"
            ].values
            if is_dead(signal):
                continue
            hist, bin_edges = np.histogram(signal, bins=50, density=True)
            mids = (bin_edges[:-1] + bin_edges[1:]) / 2

            label = "{} {}".format(pixel[0], pixel[1])
            new_fig.add_trace(go.Scatter(x=mids, y=hist, name=label))

    if plot == "PSD":
        new_fig.update_yaxes(type="log")
        new_fig.update_xaxes(type="log")
        for pixel in indexes:
            signal = ds.sel(x=pixel[0], y=pixel[1], time=slice(t_min, t_max))[
                "frames"
            ].values
            if is_dead(signal):
                continue

            freq, psd = ss.welch(signal, fs=1 / dt, nperseg=10**4)
            freq = 2 * np.pi * freq

            label = "{} {}".format(pixel[0], pixel[1])
            new_fig.add_trace(go.Scatter(x=freq, y=psd, name=label))

    return new_fig


ccf_fig.register_update_graph_callback(
    app=app, graph_id="ccf", trace_updater_id="trace-updater"
)

fig_raw.register_update_graph_callback(
    app=app, graph_id="raw", trace_updater_id="trace-updater-raw"
)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run()
