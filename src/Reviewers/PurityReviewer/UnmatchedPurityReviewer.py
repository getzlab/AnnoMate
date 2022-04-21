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
        if not os.path.exists(purity_review_session_dir):
            os.mkdir(purity_review_session_dir)
        
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
                            dfci_maf_fn_col='DFCI_local_sample_dfci_maf_fn',
                            dfci_maf_table_cols=['Hugo_Symbol', 'Chromosome', 't_alt_count', 't_ref_count', 'Tumor_Sample_Barcode']
                           ):
        app = ReviewDataApp(review_data_obj)
        
        app.add_table_from_path('DFCI MAF file', 
                                'maf-component-id', 
                                dfci_maf_fn_col, 
                                dfci_maf_table_cols)
        
        return app
    