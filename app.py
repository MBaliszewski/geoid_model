import base64
import datetime
import gpxpy
import gpxpy.gpx
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import pandas as pd
import time
from datetime import datetime
import numpy as np
from scipy.interpolate import interpn
import plotly.express as px
import requests
from io import StringIO

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

options = [{"label": "Ellipsoidal height profile", "value": "profile1"},
           {'label': 'Normal height profile', 'value': 'profile2'},
           {'label': 'Route on map', 'value': 'route'}]

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select'),
            ' .gpx file'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    dcc.Store(id='gpx_data'),

    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id='output-data-upload', style={'margin': 'auto'})),

    html.Div([
        html.Div([html.H6('Select a plot:')
                  ], style={'display': 'inline-block', 'width': '10%', 'text-align': 'center'}),

        html.Div([
            dcc.Dropdown(options=options, id='profile', value=options[0]["value"])
        ], style={'display': 'inline-block',
                  'padding-top': 5,
                  'width': '90%'})
    ], style={'display': 'flex', 'margin': 'auto'}),

    dcc.Graph(id='figure_output', style={'height': '75vh'})

], style={'margin': '20px'})


@app.callback(Output('figure_output', 'figure'),
              Input('profile', 'value'),
              Input('gpx_data', 'data'))
def update_figure(value, data):
    if data != 0:
        route = pd.read_json(data)
        if value == 'profile1':
            fig = px.line(route, x=route['cumul_dist'], y=route['elevation'],
                          custom_data=['latitude', 'longitude', 'cumul_dist', 'normal', 'elevation'])
            fig.update_layout(xaxis_title="Route meter [m]",
                              yaxis_title="Ellipsoidal height [m]",
                              xaxis=dict(
                                  tickmode='linear',
                                  tick0=min(route['cumul_dist']),
                                  dtick=2000),
                              yaxis=dict(
                                  tickmode='linear',
                                  tick0=np.round(min(route['elevation']), 0),
                                  dtick=2)
                              )

            fig.update_traces(hovertemplate='<b>Route meter: </b>%{customdata[2]:.2f} m' +
                                            '<br>Lat: %{customdata[0]}' +
                                            '<br>Lon: %{customdata[1]}' +
                                            '<br>Ellipsoidal height: %{customdata[4]:.2f} m')
            return fig
        elif value == 'profile2':
            fig = px.line(route, x=route['cumul_dist'], y=route['normal'],
                          custom_data=['latitude', 'longitude', 'cumul_dist', 'normal', 'elevation'])
            fig.update_layout(xaxis_title="Route meter [m]",
                              yaxis_title="Normal height [m]",
                              xaxis=dict(
                                  tickmode='linear',
                                  tick0=min(route['cumul_dist']),
                                  dtick=2000),
                              yaxis=dict(
                                  tickmode='linear',
                                  tick0=np.round(min(route['normal']), 0),
                                  dtick=2)
                              )
            fig.update_traces(hovertemplate='<b>Route meter: </b>%{customdata[2]:.2f} m' +
                                            '<br>Lat: %{customdata[0]}' +
                                            '<br>Lon: %{customdata[1]}' +
                                            '<br>Normal height: %{customdata[3]:.2f} m')
            return fig
        elif value == 'route':
            fig = px.line_mapbox(route, lat="latitude", lon="longitude", zoom=10,
                                 custom_data=['latitude', 'longitude', 'cumul_dist', 'normal', 'elevation'])
            fig.update_traces(mode="markers+lines")
            fig.update_traces(hovertemplate='<b>Route meter: </b>%{customdata[2]:.2f} m' +
                                            '<br>Lat: %{customdata[0]}' +
                                            '<br>Lon: %{customdata[1]}' +
                                            '<br>Normal height: %{customdata[3]:.2f} m' +
                                            '<br>Ellipsoidal height: %{customdata[4]:.2f} m')
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0})
            return fig
    else:
        return {'data': []}


def parse_contents(filename, date):
    name = filename

    if name is not None:
        d = datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")
        return html.Div([
            html.H5(f'Filename: {name}'),
            html.H6(f'Created: {d}'),
        ])
    else:
        return html.Div([
            html.H5(f'Filename:'),
            html.H6(f'Created:'),
        ])


@app.callback(Output('output-data-upload', 'children'),
              Output('gpx_data', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        children = [parse_contents(None, None)]
        return children, 0
    elif len(list_of_contents) != 1:
        n_files = len(list_of_contents)
        return html.H6(f'{n_files} files loaded, but 1 file are required'), 0
    else:
        children = [
            parse_contents(n, d) for n, d in
            zip(list_of_names, list_of_dates)]

        pf = all_data(list_of_contents[0])
        pf = pf.to_json(orient='columns')
        time.sleep(0)
        return children, pf


def xyz(phi, lam, h):
    a = 6378137  # wielka półoś dla elipsoidy GRS80
    e = 0.00669438002290  # kwadrat pierwszego mimośrodu dla elipsoidy GRS80
    n = a / (np.sqrt(1 - e * np.power(np.sin(phi), 2)))  # promień przekroju Ziemi w kierunku I Wertykału
    x = (n + h) * np.cos(phi) * np.cos(lam)
    y = (n + h) * np.cos(phi) * np.sin(lam)
    z = (n * (1 - e) + h) * np.sin(phi)
    return [x, y, z]


def all_data(route):
    # konwersja wczytanego pliku
    content_type, content_string = route.split(',')
    decoded_route = base64.b64decode(content_string)

    # odczyt pliku gpx
    gpx = gpxpy.parse(decoded_route)
    route_info = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_info.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time
                })

    route_df = pd.DataFrame(route_info)     # data frame z danymi z pliku gpx

    # obliczenie odległości między pomierzonymi punktami
    dist = [0]
    for i in range(len(route_df)):
        if i == 0:
            continue
        else:
            phi1 = route_df.iloc[i - 1]['latitude']
            lam1 = route_df.iloc[i - 1]['longitude']
            h1 = route_df.iloc[i - 1]['elevation']
            xyz1 = xyz(np.deg2rad(phi1), np.deg2rad(lam1), h1)

            phi2 = route_df.iloc[i]['latitude']
            lam2 = route_df.iloc[i]['longitude']
            h2 = route_df.iloc[i]['elevation']
            xyz2 = xyz(np.deg2rad(phi2), np.deg2rad(lam2), h2)

            d = np.linalg.norm(np.array(xyz2) - np.array(xyz1))
            dist.append(d)

    route_df['distance'] = dist     # kolumna z odległościami między parami sąsiednich punktów
    route_df['cumul_dist'] = route_df['distance'].cumsum()  # kolumna z zsumowanymi odległościami od początku trasy

    # wczytanie modelu geoidy
    model1 = np.genfromtxt('x00.txt', skip_header=1)
    model2 = np.genfromtxt('x01.txt', skip_header=1)
    model = np.concatenate((model1, model2), axis=0)

    # z modelu geoidy utworzenie tablic, których można użyć w funkcji do interpolacji
    x = model[:, 0]
    y = model[:, 1]
    z = model[:, 2]
    x_grid = x.reshape((len(np.unique(x)), -1))
    y_grid = y.reshape((len(np.unique(x)), -1))
    z_grid = z.reshape((len(np.unique(x)), -1))

    x_range = x_grid[:, 0]
    y_range = y_grid[0, :]

    # interpolacja każdego punktu trasy, aby otrzymać zeta w danym punkcie trasy
    zeta_all = []
    for i in range(len(route_df)):
        fi = route_df.iloc[i]['latitude']
        lam = route_df.iloc[i]['longitude']
        zeta = interpn((x_range, y_range), z_grid, (fi, lam))
        zeta_all.append(float(zeta))

    # nowa kolumna z wysokościami normalmnymi
    route_df['normal'] = route_df['elevation'] - np.array(zeta_all)

    return route_df


if __name__ == '__main__':
    app.run_server(debug=True)
