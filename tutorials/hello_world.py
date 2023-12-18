from dash import Dash, Input, Output, callback, dcc, html
import copy
import plotly.express as px
import pandas as pd

import plotly.figure_factory as ff
import plotly.graph_objects as go

import numpy as np

x, y = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25))
z = x*np.exp(-x**2 - y**2)
conf = z ** 2 + 2
v, u = np.gradient(z, .2, .2)

# Create quiver figure
fig = ff.create_quiver(x, y, u, v,
                       scale=.25,
                       arrow_scale=.4,
                       name='quiver',
                       line_width=1,
                       hovertemplate='conf: %{conf:.2f}')

# Add points to figure
fig.add_trace(go.Scatter(x=[-.7, .75], y=[0,0],
                    mode='markers',
                    marker_size=12,
                    name='points'))


app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Input(id="x1", type="number", placeholder="", value=0, style={'marginRight': '10px'}),
        dcc.Graph(id="ccf", figure={}),
        dcc.Graph(id="quiver", figure=fig, style={'width': '90vh', 'height': '90vh'})
    ]
)

@callback(
    Output("ccf", "figure"),
    Input("x1", "value"),
)
def update_output(x1):
    y_vals = np.arange(-2, 2, .2)
    figg = px.line(dict(y=y_vals, u=u[x1, :]), x="y", y="u")
    return figg


if __name__ == "__main__":
    app.run()


