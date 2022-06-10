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
    def gen_review_app(self, test_param) -> ReviewDataApp:
        app = ReviewDataApp()
        default_maf_cols = [
            'Hugo_Symbol',
            'Chromosome',
            'Start_position',
            'Protein_change',
            'Variant_Classification',
            't_ref_count',
            't_alt_count',
        ]
        maf_cols_options = []
        maf_cols_value = []

        def gen_clinical_data_table(df, idx, cols):
            r=df.loc[idx]
            return [dbc.Table.from_dataframe(r[cols].to_frame().reset_index())]

        app.add_component(AppComponent(
            'Clinical Data',
            html.Div(
                dbc.Table.from_dataframe(df=pd.DataFrame()),
                id='clinical-data-component'
            ),
            callback_output=[Output('clinical-data-component', 'children')],
            new_data_callback=gen_clinical_data_table
        ), cols=['participant_id', 'gender', 'age_at_diagnosis', 'vital_status', 'death_date_dfd'])

        def gen_maf_columns(df, idx, cols):
            maf_df = pd.read_csv(df.loc[idx, 'phylogic_all_pairs_mut_ccfs'], sep='\t')
            maf_cols_options = (list(maf_df))

            for col in default_maf_cols:
                if col in maf_cols_options and col not in maf_cols_value:
                    maf_cols_value.append(col)

            for col in cols:
                if col in maf_cols_options and col not in maf_cols_value:
                    maf_cols_value.append(col)

            return [
                maf_df,
                maf_cols_options,
                maf_cols_value
            ]

        def gen_maf_table(df, idx, cols):
            maf_df, maf_cols_options, maf_cols_value = gen_maf_columns(df, idx, cols)

            return [
                maf_cols_options,
                maf_cols_value,
                dash_table.DataTable(
                        data=pd.read_csv(df.loc[idx, 'phylogic_all_pairs_mut_ccfs'], sep='\t').to_dict('records'),
                        columns=[{'name': i, 'id': i, 'selectable': True} for i in maf_cols_value],
                        filter_action='native',
                        row_selectable='multi',
                        column_selectable='multi',
                        page_action='native',
                        page_current=0,
                        page_size=10
                )
            ]

        def internal_gen_maf_table(df, idx, cols):
            maf_df, maf_cols_options, maf_cols_value = gen_maf_columns(df, idx, cols)

            return [
                maf_cols_options,
                cols,
                dash_table.DataTable(
                        data=pd.read_csv(df.loc[idx, 'phylogic_all_pairs_mut_ccfs'], sep='\t').to_dict('records'),
                        columns=[{'name': i, 'id': i, 'selectable': True} for i in cols],
                        filter_action='native',
                        row_selectable='multi',
                        column_selectable='multi',
                        page_action='native',
                        page_current=0,
                        page_size=10
                )
            ]

        app.add_component(AppComponent(
            'Mutations',
            html.Div([
                html.Div(dcc.Dropdown(
                    maf_cols_options,
                    maf_cols_value,
                    multi=True,
                    id='column-selection-dropdown'
                )),

                html.Div(dash_table.DataTable(
                    columns=[{'name': i, 'id': i, 'selectable': True} for i in pd.DataFrame().columns],
                    data=pd.DataFrame().to_dict('records'),
                    id='mutation-table'
                ), id='mutation-table-component')
            ]),

            callback_input=[Input('column-selection-dropdown', 'value')],
            callback_output=[
                Output('column-selection-dropdown', 'options'),
                Output('column-selection-dropdown', 'value'),
                Output('mutation-table-component', 'children')
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
