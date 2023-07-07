"""
DataTableComponents module contains methods to generate components for displaying table information
"""
import pandas as pd
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.stats import beta
import pickle

from AnnoMate.ReviewDataApp import AppComponent
from AnnoMate.AppComponents.utils import cluster_color, get_unique_identifier
from AnnoMate.DataTypes.GenericData import GenericData


def gen_console_links(gsurl: str, link_display_name=None):
    link_display_name = gsurl if link_display_name is None else link_display_name
    
    return html.A(
        html.P(link_display_name),
        href=f"https://console.cloud.google.com/storage/browser/_details/{gsurl.split('/', 2)[-1]}",
        target="_blank"
    )

def gen_annotated_data_info_table(
    data: GenericData,
    data_id,
    cols,
    data_attribute,
    generate_console_links=True,
    console_link_col_name='Console_link',
    link_display_name='Click here',
):
    """
    generate_console_links: Whether you want to convert gsurls to console links
    console_link_col_name: specify the column name where the links will be displayed
    """
    data_df = getattr(data, data_attribute)
    
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError(f'data_attribution {data_attribute} from data object is not a pandas dataframe.')
        
    if data_df.index.tolist() != data.annot_df.index.tolist():
        raise ValueError(f'data.{data_attribute} index does not match the data.annot_df index.')
    
    r = data_df.loc[data_id]
    tmp_data_df = r[cols].to_frame()
    tmp_data_df[r.name] = tmp_data_df[r.name].astype(str)
            
    if generate_console_links:
        
        tmp_data_df[console_link_col_name] = tmp_data_df[r.name].apply(
            lambda url: gen_console_links(url, link_display_name) if 'gs://' in url else ''
        )

    return [[html.H2(f'{r.name} Data Summary'), dbc.Table.from_dataframe(tmp_data_df.reset_index())]]

def gen_annotated_data_info_table_layout(table_id):
    return [
        html.Div(
            children=[
                html.H2('Data Summary'),
                dbc.Table.from_dataframe(df=pd.DataFrame())
            ],
            id=table_id
        )
    ]

def gen_annotated_data_info_table_component(
    component_name='Annotated Data Information table',
    table_id='annot-data-info-component'
):
    """
    Generate a non-interactive table related to the subjects you are reviewing. 
    The table can include hyperlinks to any data that references a gsurl (ie gs://). 
    See gen_annotated_data_info_table() for details.
    """
    return AppComponent(
        component_name,
        layout=gen_annotated_data_info_table_layout(table_id=table_id),
        callback_output=[Output(table_id, 'children')],
        new_data_callback=gen_annotated_data_info_table
    )
    
