from getzlab_JupyterReviewer.src.ReviewData import ReviewData, ReviewDataAnnotation
from getzlab_JupyterReviewer.src.ReviewDataApp import ReviewDataApp

import pandas as pd
import numpy as np
import functools
import time

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

from getzlab_JupyterReviewer.src.Reviewers.Reviewer import Reviewer
from getzlab_JupyterReviewer.src.lib.plot_cnp import plot_acr_interactive

from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
import os
import pickle

class UnmatchedPurityReviewer(Reviewer):


    def gen_review_data_object(session_dir, 
                               df: pd.DataFrame, 
                               more_annot_cols: {str: ReviewDataAnnotation}):
        purity_review_session_dir = f'{session_dir}/purity_review_session'
        
        annot_data = {'purity': ReviewDataAnnotation( 'number', 
                                           validate_input=lambda x: (x <= 1.0) and (x >= 0.0)),
                      'ploidy': ReviewDataAnnotation('number', 
                                           validate_input=lambda x: x >= 0.0)}
        
        rd = ReviewData(review_dir=purity_review_session_dir,
                        df=df, # optional if directory above already exists. 
                        annotate_data = {**annot_data, **more_annot_cols})
        
        print(f'Created purity review session in {purity_review_session_dir}')
        return rd
    
    
    def gen_review_data_app(review_data_obj: ReviewData,
                            sample_table_cols,
                           ):
        app = ReviewDataApp(review_data_obj)
        
        def gen_data_summary_table(r, cols):
            sample_data_df = r[cols].to_frame()
            sample_data_df[r.name] = sample_data_df[r.name].astype(str)
            sample_data_df['Console_link'] = ''
            for attr, value in sample_data_df.iterrows():
                if 'gs://' in value[r.name]:
                    path = value[r.name].split('/', 2)[-1]
                    sample_data_df.loc[attr, 'Console_link'] = f"https://console.cloud.google.com/storage/browser/_details/{path}"
            sample_data_df['Console_link'] = sample_data_df['Console_link'].apply(lambda url: html.A(html.P(url),
                                                                                      href=url,
                                                                                      target="_blank"))
            return [[html.H1(f'{r.name} Data Summary'), dbc.Table.from_dataframe(sample_data_df.reset_index())]]

        app.add_custom_component('sample-info-component', 
                                  html.Div(children=[html.H1('Data Summary'), 
                                                     dbc.Table.from_dataframe(df=pd.DataFrame())],
                                           id='sample-info-component'
                                          ), 
                                  callback_output=[Output('sample-info-component', 'children')],
                                  func=gen_data_summary_table, 
                                  cols=sample_table_cols)

        
        return app
    