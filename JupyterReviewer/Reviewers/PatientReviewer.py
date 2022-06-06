import pandas as pd
import numpy as np
import functools
import time
import os

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
        reuse_existing_review_data_fn: str = None):

        review_data_annotation_dict = {
            'purity': ReviewDataAnnotation('number', validate_input=validate_purity),
            'ploidy': ReviewDataAnnotation('number', validate_input=validate_ploidy),
            'rating': ReviewDataAnnotation('number', options=range(10)),
            'description': ReviewDataAnnotation('text'),
            'class': ReviewDataAnnotation('radioitem', options=[f'Option {n}' for n in range(4)])
        }

        rd = ReviewData(
            review_data_fn=review_data_fn,
            description=description,
            df=df,
            review_data_annotation_dict = review_data_annotation_dict
        )

    def gen_review_app(self, test_param) -> ReviewDataApp:
        app = ReviewDataApp()

        # app.add_table_from_path(
        #     table_title='Phylogic Mutation CCFs',
        #     component_id='phylogic-component-id',
        #     table_fn_col='phylogic_all_pairs_mut_ccfs',
        #     table_cols=[
        #         'Sample_ID',
        #         'Hugo_Symbol',
        #         'Chromosome',
        #         'Reference_Allele',
        #         'Tumor_Seq_Allele',
        #         'Variant_Classification',
        #         'Variant_Type',
        #         'preDP_ccf_mean',
        #         'clust_ccf_mean'
        #     ]
        # )

        phylogic_df = pd.read_csv(f'/Users/svanseters/Broad/JupyterReviewer/example_notebooks/example_data/ccf.txt', sep='\t', usecols=[
            'Sample_ID',
            'Hugo_Symbol',
            'Chromosome',
            'Reference_Allele',
            'Tumor_Seq_Allele',
            'Variant_Classification',
            'preDP_ccf_mean',
            'clust_ccf_mean'
        ])

        app.add_component(AppComponent(
            name='Phylogic Data',
            layout=html.Div([
                html.P('Filter by variant by typing in the variant you want to focus on. Filter by chromosome by typing =<chromosome>. Filter by ccf using <, <=, >, >=, =.'),
                dash_table.DataTable(
                    id='phy-table',
                    columns=[{'name': i, 'id': i} for i in phylogic_df.columns],
                    data=phylogic_df.to_dict('records'),
                    filter_action='native'
                )
            ])
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
