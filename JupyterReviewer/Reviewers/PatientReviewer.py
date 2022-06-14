import pandas as pd
import numpy as np
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
import functools

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
#from JupyterReviewer.lib.plot_cnp import plot_acr_interactive

def validate_purity(x):
    return (x >= 0) and (x <= 1)

def validate_ploidy(x):
    return x >= 0

def gen_clinical_data_table(df, idx, cols):
    r=df.loc[idx]
    return [dbc.Table.from_dataframe(r[cols].to_frame().reset_index())]

default_maf_cols = [
    'Hugo_Symbol',
    'Chromosome',
    'Start_position',
    'End_position'
    'Protein_change',
    'Variant_Classification',
    't_ref_count',
    't_ref_count_pre_forecall',
    't_alt_count',
    't_alt_count_pre_forecall',
    'n_ref_count',
    'n_alt_count'
]
maf_cols_options = []
maf_cols_value = []
hugo_symbols = []
variant_classifications = []
cluster_assignments = [1, 2, 3, 4, 5, 6, 7, 8, 9]

cluster_assignment_colors_dict = {
    1: 'OliveDrab',
    2: 'LightSkyBlue',
    3: 'GoldenRod',
    4: 'DimGray',
    5: 'MediumSlateBlue',
    6: 'Maroon',
    7: 'MediumAquaMarine',
    8: 'LightPink',
    9: 'RebeccaPurple'
}

def style_data_format(column_id, filter_query, color='Black', backgroundColor='White'):
    return {
        'if': {
            'column_id': column_id,
            'filter_query': '{%s} = "%s"' % (column_id, filter_query)
        },
        'color': color,
        'backgroundColor': backgroundColor,
        'fontWeight': 'bold'
    }

def gen_style_data_conditional():
    style_data_conditional = []

    if 'Cluster_Assignment' in maf_cols_value:
        for n in range(1, 10):
            style_data_conditional.append(style_data_format('Cluster_Assignment', n, color=cluster_assignment_colors_dict[n]))

    if 'functional_effect' in maf_cols_value:
        style_data_conditional.extend([
            style_data_format('functional_effect', 'Likely Loss-of-function', backgroundColor='DarkOliveGreen'),
            style_data_format('functional_effect', 'Likely Gain-of-function', backgroundColor='DarkSeaGreen')
        ])

    if 'oncogenic' in maf_cols_value:
        style_data_conditional.append(style_data_format('oncogenic', 'Likely Oncogenic', backgroundColor='DarkOliveGreen'))

    if 'dbNSFP_Polyphen2_HDIV_ann' in maf_cols_value:
        style_data_conditional.append(style_data_format('dbNSFP_Polyphen2_HDIV_ann', 'D', backgroundColor='FireBrick'))

    return style_data_conditional

def gen_maf_columns(df, idx, cols, hugo, variant, cluster):
    #maf_df = pd.read_csv(df.loc[idx, 'phylogic_all_pairs_mut_ccfs'], sep='\t')
    maf_df = pd.read_csv('~/Broad/JupyterReviewer/example_notebooks/example_data/all_mut_ccfs_maf_annotated_w_cnv_single_participant.txt', sep='\t')
    maf_cols_options = (list(maf_df))

    for col in default_maf_cols:
        if col in maf_cols_options and col not in maf_cols_value:
            maf_cols_value.append(col)

    for col in cols:
        if col in maf_cols_options and col not in maf_cols_value:
            maf_cols_value.append(col)

    for symbol in maf_df.Hugo_Symbol.unique():
        hugo_symbols.append(symbol)

    for classification in maf_df.Variant_Classification.unique():
        variant_classifications.append(classification)

    filtered_maf_df = maf_df.copy()
    if hugo:
        filtered_maf_df = filtered_maf_df[filtered_maf_df.Hugo_Symbol.isin(hugo)]
    if variant:
        filtered_maf_df = filtered_maf_df[filtered_maf_df.Variant_Classification.isin(variant)]
    if cluster:
        filtered_maf_df = filtered_maf_df[filtered_maf_df.Cluster_Assignment.isin(cluster)]

    return [
        maf_df,
        maf_cols_options,
        maf_cols_value,
        hugo_symbols,
        variant_classifications,
        filtered_maf_df
    ]

def gen_maf_table(df, idx, cols, hugo, table_size, variant, cluster):
    maf_df, maf_cols_options, maf_cols_value, hugo_symbols, variant_classifications, filtered_maf_df = gen_maf_columns(df, idx, cols, hugo, variant, cluster)

    return [
        maf_cols_options,
        maf_cols_value,
        dash_table.DataTable(
            data=filtered_maf_df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'selectable': True} for i in maf_cols_value],
            filter_action='native',
            row_selectable='single',
            column_selectable='multi',
            page_action='native',
            page_current=0,
            page_size=table_size,
            style_data_conditional=gen_style_data_conditional()
        ),
        hugo_symbols,
        variant_classifications
    ]

def internal_gen_maf_table(df, idx, cols, hugo, table_size, variant, cluster):
    maf_df, maf_cols_options, maf_cols_value, hugo_symbols, variant_classifications, filtered_maf_df = gen_maf_columns(df, idx, cols, hugo, variant, cluster)

    return [
        maf_cols_options,
        cols,
        dash_table.DataTable(
                data=filtered_maf_df.to_dict('records'),
                columns=[{'name': i, 'id': i, 'selectable': True} for i in cols],
                filter_action='native',
                row_selectable='single',
                column_selectable='multi',
                page_action='native',
                page_current=0,
                page_size=table_size,
                style_data_conditional=gen_style_data_conditional()
        ),
        hugo_symbols,
        variant_classifications
    ]

class PatientReviewer(ReviewerTemplate):

    def gen_review_data(
        self,
        review_data_fn: str,
        description: str='',
        df: pd.DataFrame = pd.DataFrame(),
        review_data_annotation_dict: {str: ReviewDataAnnotation} = {},
        reuse_existing_review_data_fn: str = None
    ):

        review_data_annotation_dict = {
            'purity': ReviewDataAnnotation('number', validate_input=validate_purity),
            'ploidy': ReviewDataAnnotation('number', validate_input=validate_ploidy),
            'mutation': ReviewDataAnnotation('text'),
            'class': ReviewDataAnnotation('radioitem', options=['Possible Driver', 'Likely Driver', 'Possible Artifact', 'Likely Artifact']),
            'description': ReviewDataAnnotation('text')
        }

        rd = ReviewData(
            review_data_fn=review_data_fn,
            description=description,
            df=df,
            review_data_annotation_dict = review_data_annotation_dict
        )

        return rd

    # list optional cols param
    def gen_review_app(self) -> ReviewDataApp:
        app = ReviewDataApp()

        app.add_component(AppComponent(
            'Clinical Data',
            html.Div(
                dbc.Table.from_dataframe(df=pd.DataFrame()),
                id='clinical-data-component'
            ),
            callback_output=[Output('clinical-data-component', 'children')],
            new_data_callback=gen_clinical_data_table
        ), cols=['participant_id', 'gender', 'age_at_diagnosis', 'vital_status', 'death_date_dfd'])

        app.add_component(AppComponent(
            'Mutations',
            html.Div([
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
                                options=[10,20,30],
                                value=10,
                                id='table-size-dropdown'
                            )
                        ], width=2),
                        dbc.Col([
                            dcc.Dropdown(
                                maf_cols_options,
                                maf_cols_value,
                                multi=True,
                                id='column-selection-dropdown'
                            )
                        ], width=10)
                    ])
                ]),

                html.Div(
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                options=hugo_symbols,
                                multi=True,
                                placeholder='Filter by Hugo Symbol',
                                id='hugo-dropdown'
                            )
                        ], width=2),
                        dbc.Col([
                            dcc.Dropdown(
                                options=variant_classifications,
                                multi=True,
                                placeholder='Filter by Variant Classification',
                                id='variant-classification-dropdown'
                            )
                        ], width=2),
                        dbc.Col([
                            dcc.Dropdown(
                                options=cluster_assignments,
                                multi=True,
                                placeholder='Filter by Cluster Assignment',
                                id='cluster-assignment-dropdown'
                            )
                        ], width=2)
                    ])
                ),

                html.Div(dash_table.DataTable(
                    columns=[{'name': i, 'id': i, 'selectable': True} for i in pd.DataFrame().columns],
                    data=pd.DataFrame().to_dict('records'),
                    id='mutation-table'
                ), id='mutation-table-component'),
            ]),

            callback_input=[
                Input('column-selection-dropdown', 'value'),
                Input('hugo-dropdown', 'value'),
                Input('table-size-dropdown', 'value'),
                Input('variant-classification-dropdown', 'value'),
                Input('cluster-assignment-dropdown', 'value')
            ],
            callback_output=[
                Output('column-selection-dropdown', 'options'),
                Output('column-selection-dropdown', 'value'),
                Output('mutation-table-component', 'children'),
                Output('hugo-dropdown', 'options'),
                Output('variant-classification-dropdown', 'options')
            ],
            new_data_callback=gen_maf_table,
            internal_callback=internal_gen_maf_table
        ))

        app.add_component(AppComponent(
            'Purity Slider',
            html.Div(dcc.Slider(0, 1, 0.1, value=0.5, id='a-slider')),
            callback_output=[Output('a-slider', 'value')],
            callback_states_for_autofill=[State('a-slider', 'value')]
        ))

        return app

    def gen_autofill(self):
        self.add_autofill('Purity Slider', {'purity': State('a-slider', 'value')})
