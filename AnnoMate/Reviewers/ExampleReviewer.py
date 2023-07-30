"""Example Reviewer Description
A basic reviewer for the AnnoMate tutorial.
Uses simulated data from simulated_data directory
"""
from AnnoMate.Data import DataAnnotation
from AnnoMate.DataTypes.GenericData import GenericData
from AnnoMate.ReviewDataApp import ReviewDataApp, AppComponent
from AnnoMate.AnnotationDisplayComponent import *
from AnnoMate.ReviewerTemplate import ReviewerTemplate

import pandas as pd
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from typing import Dict, List
import plotly.express as px
import dash_bootstrap_components as dbc


# For pickling to work, need to explicitly define function
def validate_purity(x):
    return x < 0.5


class ExampleReviewer(ReviewerTemplate):
    def gen_data(self,
                 description: str,
                 annot_df: pd.DataFrame,
                 annot_col_config_dict: Dict,
                 history_df: pd.DataFrame,
                 sample_df: pd.DataFrame,
                 preprocessing_str: str,
                 index: List = None,
                 ) -> GenericData:
        """
        Creates a dataframe with index corresponding to the primary dataframe's index

        Parameters
        ----------
        sample_df: pd.DataFrame
            dataframe with samples data and sample ids in the index
        preprocessing_str: str
            Example of preprocessing data
        """
        sample_df['new_column'] = preprocessing_str

        if index is None:
            index = sample_df.index.tolist()

        return GenericData(index=index,
                           description=description,
                           df=sample_df,
                           annot_df=annot_df,
                           annot_col_config_dict=annot_col_config_dict,
                           history_df=history_df
                           )

    def set_default_review_data_annotations(self):
        self.add_review_data_annotation('Notes', DataAnnotation('string'))
        self.add_review_data_annotation('Flag', DataAnnotation('string', options=['Keep', 'Remove'], default='Keep'))

    def gen_review_app(self, mut_file_col: str, sample_cols: List[str]) -> ReviewDataApp:
        """
        Example ReviewDataApp

        Parameters
        ----------
        mut_file_col: str
            column in data object's df dataframe that contains file paths to mutation file for current sample
        sample_cols: List[str]
            List of columns in data object's df dataframe to display per sample being reviewed
        """
        app = ReviewDataApp()

        def gen_data_summary_table(data: GenericData, idx, cols):
            df = data.df
            r = df.loc[idx]
            return [[html.H1(f'{r.name} Data Summary'), dbc.Table.from_dataframe(r[cols].to_frame().reset_index())]]

        app.add_component(AppComponent(name='sample-info-component',
                                       layout=html.Div(children=[html.H1('Data Summary'),
                                                                 dbc.Table.from_dataframe(df=pd.DataFrame())],
                                                       id='sample-info-component'
                                                       ),
                                       callback_output=[Output('sample-info-component', 'children')],
                                       new_data_callback=gen_data_summary_table,
                                       ),
                          cols=sample_cols
                          )

        app.add_table_from_path(data_table_source='df',
                                table_title='Maf file',
                                component_id='maf-component-id',
                                table_fn_col=mut_file_col,
                                table_cols=['sample_id', 'gene', 'vaf', 'cov',
                                            't_alt_count', 't_ref_count'])

        def plot_muts(data: GenericData, idx, color, mut_file_col):
            mut_fn = data.df.loc[idx, mut_file_col]
            maf_df = pd.read_csv(mut_fn, sep='\t')
            fig = px.histogram(maf_df, x='vaf', color_discrete_sequence=[color])
            fig.update_traces()
            return [fig]

        app.add_component(AppComponent(name='Mut vafs',
                                       layout=html.Div([
                                           dcc.Graph(figure={}, id='mut-figure'),
                                           dbc.RadioItems(options=[{'label': c, 'value': c} for c in ['red',
                                                                                                      'blue',
                                                                                                      'green']],
                                                          id='mut-figure-color-radioitem',
                                                          value='red'
                                                          )]),
                                       callback_output=[Output('mut-figure', 'figure')],
                                       callback_input=[Input('mut-figure-color-radioitem', 'value')],
                                       new_data_callback=plot_muts,
                                       internal_callback=plot_muts,
                                       ),
                          mut_file_col=mut_file_col
                          )

        return app

    def set_default_review_data_annotations_app_display(self):
        self.add_annotation_display_component('Notes', TextAreaAnnotationDisplay())
        self.add_annotation_display_component('Flag', RadioitemAnnotationDisplay())

    def set_default_autofill(self):
        self.add_autofill(
            autofill_button_name='example-autofill-button', 
            fill_value='Nothing to say', 
            annot_name='Notes'
        )
