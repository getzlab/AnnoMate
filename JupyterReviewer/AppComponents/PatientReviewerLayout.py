import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools
import time
import os
import re

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import Dash, dash_table
import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import functools
import itertools

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate


class PatientReviewerLayout:

    def gen_mutation_table_layout():
        return html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P('Table Size (Rows)'),
                    ], width=2),
                    dbc.Col([
                        html.P('Select Columns')
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='table-size-dropdown',
                            options=[10,20,30],
                            value=10
                        )
                    ], width=2),
                    dbc.Col([
                        dcc.Dropdown(
                            id='column-selection-dropdown',
                            options=[],
                            value=[],
                            multi=True,
                        )
                    ], width=10)
                ])
            ]),

            html.Div(
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='hugo-dropdown',
                            options=[],
                            multi=True,
                            placeholder='Filter by Hugo Symbol',
                        )
                    ], width=2),
                    dbc.Col([
                        dcc.Dropdown(
                            id='variant-classification-dropdown',
                            options=[],
                            multi=True,
                            placeholder='Filter by Variant Classification'
                        )
                    ], width=2),
                    dbc.Col([
                        dcc.Dropdown(
                            id='cluster-assignment-dropdown',
                            options=[],
                            multi=True,
                            placeholder='Filter by Cluster Assignment'
                        )
                    ], width=2)
                ])
            ),

            html.Div(dash_table.DataTable(
                id='mutation-table',
                columns=[{'name': i, 'id': i, 'selectable': True} for i in pd.DataFrame().columns],
                data=pd.DataFrame().to_dict('records')
            ), id='mutation-table-component'),
        ])

    def gen_phylogic_components_layout():
        return html.Div([
            dcc.Checklist(
                id='time-scale-checklist',
                options=['Time Scaled'],
                value=['Time Scaled'],
            ),
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Graph(
                                id='ccf-plot',
                                figure=go.Figure()
                            ),
                        ])
                    ], width=8, align='center'),
                    dbc.Col([
                        html.Div(
                            cyto.Cytoscape(
                                id='phylogic-tree',
                                elements=[],
                                style={'width': '100%', 'height': '450px'},
                            ),
                            id='phylogic-tree-component'
                        ),
                        dcc.Dropdown(
                            id='tree-dropdown',
                            options=[]
                        )
                    ], width=4, align='center')
                ])
            ])
        ])

    def gen_cnv_plot_layout():
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='cnv_plot',
                        figure=go.Figure()
                    ),
                ], width=10),
                dbc.Col([
                    html.H3('Customize Plot'),
                    html.H5('Samples:'),
                    dcc.Checklist(
                        id='sample-selection-checklist',
                        options=[],
                        value=[],
                        labelStyle={'display': 'block'}
                    ),
                    html.P(''),
                    html.H5('Sigmas:'),
                    dcc.Checklist(
                        id='sigma_checklist',
                        options=['Show CNV Sigmas'],
                        value=['Show CNV Sigmas']
                    ),
                    html.P(''),
                    html.H5('Colors:'),
                    dcc.RadioItems(
                        id='cnv-color-radioitem',
                        options=['Differential', 'Cluster', 'Red/Blue', 'Clonal/Subclonal'],
                        value='Differential',
                        labelStyle={'display': 'block'}
                    ),
                    html.P(''),
                    html.H5('Scale:'),
                    dcc.Checklist(
                        options=['Display Absolute CN'],
                        value=['Display Absolute CN'],
                        id='absolute-cnv-box'
                    ),
                    html.P(''),
                    html.Button('Submit', id='cnv-button')
                ], width=2)
            ]),
        ])
