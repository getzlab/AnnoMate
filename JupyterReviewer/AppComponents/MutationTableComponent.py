"""MutationTableComponent.py module

Interactive Mutation Table with column selection, sorting, selecting, and filtering

"""

import pandas as pd
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import Dash, dash_table
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from JupyterReviewer.ReviewDataApp import AppComponent
from JupyterReviewer.AppComponents.utils import cluster_color, get_unique_identifier
from JupyterReviewer.DataTypes.PatientSampleData import PatientSampleData


def gen_mutation_table_app_component():
    """Generate Interactive Mutation Table component"""
    return AppComponent(
        'Mutations',
        layout=gen_mutation_table_layout(),

        callback_input=[
            Input('column-selection-dropdown', 'value'),
            Input('hugo-dropdown', 'value'),
            Input('table-size-dropdown', 'value'),
            Input('variant-classification-dropdown', 'value'),
            Input('cluster-assignment-dropdown', 'value'),
            # Input('mutation-table', 'selected_row_ids'),  # selected rows regardless of filtering
            # Input('mutation-table', 'derived_virtual_row_ids')  # all rows in table after filtering (and sorting)
        ],
        callback_output=[
            Output('column-selection-dropdown', 'options'),
            Output('column-selection-dropdown', 'value'),
            Output('mutation-table-component', 'children'),
            Output('hugo-dropdown', 'options'),
            Output('variant-classification-dropdown', 'options'),
            Output('cluster-assignment-dropdown', 'options'),
            Output('mutation-sample-table-component', 'children')
        ],
        new_data_callback=gen_maf_table,
        internal_callback=internal_gen_maf_table
    )

def gen_mutation_table_layout():
    """Generate Mutation Table component layout"""
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

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dash_table.DataTable(
                            id='mutation-table',
                            columns=[{'name': i, 'id': i, 'selectable': True} for i in pd.DataFrame().columns],
                            data=pd.DataFrame().to_dict('records')
                        ), id='mutation-table-component'
                    )
                ], width=7),
                dbc.Col([
                    html.Div(
                        dash_table.DataTable(
                            id='mutation-sample-table',
                            columns=[{'name': i, 'id': i, 'selectable': True} for i in pd.DataFrame().columns],
                            data=pd.DataFrame().to_dict('records')
                        ), id='mutation-sample-table-component'
                    )
                ], width=5),
            ])
        ])
    ])

def format_style_data(column_id, filter_query, color='Black', backgroundColor='White'):
    """Formatting for mutation table coloring

    Parameters
    ----------
    column_id
        name of the column that the content to be colored is in
    filter_query
        content to be colored
    color
        text color
    backgroundColor

    Returns
    -------
    dict
        dict following the style_data_conditinal format for a dash DataTable

    """
    return {
        'if': {
            'column_id': column_id,
            'filter_query': '{%s} = "%s"' % (column_id, filter_query)
        },
        'color': color,
        'backgroundColor': backgroundColor,
        'fontWeight': 'bold'
    }

def gen_style_data_conditional(maf_df, custom_colors, maf_cols_value):
    """Generate mutation table coloring and add custom colors if given.

    Parameters
    ----------
    maf_df
        maf file DataFrame
    custom_colors
        custom_colors kwarg from gen_review_app

    Returns
    -------
    style_data_conditinal: list of dicts
        list of dicts from format_style_data

    """
    style_data_conditional = []

    if 'Cluster_Assignment' in maf_cols_value:
        for n in maf_df.Cluster_Assignment.unique():
            style_data_conditional.append(format_style_data('Cluster_Assignment', n, color=cluster_color(n)))

    if 'functional_effect' in maf_cols_value:
        style_data_conditional.extend([
            format_style_data('functional_effect', 'Likely Loss-of-function', backgroundColor='DarkOliveGreen'),
            format_style_data('functional_effect', 'Likely Gain-of-function', backgroundColor='DarkSeaGreen')
        ])

    if 'oncogenic' in maf_cols_value:
        style_data_conditional.append(format_style_data('oncogenic', 'Likely Oncogenic', backgroundColor='DarkOliveGreen'))

    if 'dbNSFP_Polyphen2_HDIV_ann' in maf_cols_value:
        style_data_conditional.append(format_style_data('dbNSFP_Polyphen2_HDIV_ann', 'D', backgroundColor='FireBrick'))

    if custom_colors:
        for list in custom_colors:
            style_data_conditional.append(format_style_data(list[0], list[1], list[2], list[3]))

    return style_data_conditional

def gen_maf_columns(df, idx, cols, hugo, variant, cluster):
    """Generate mutation table columns from selected columns and filtering dropdowns.

    Parameters
    ----------
    df
        participant level DataFrame
    idx
    cols
        column selection dropdown value
    hugo
        hugo symbol filtering dropdown value
    variant
        variant classification filtering dropdown value
    cluster
        cluster assignment filtering dropdown value

    Returns
    -------
    maf_cols_options : list of str
        options in mutation table columns dropdown
    maf_cols_value : list of str
        values selected in mutation table columns dropdown
    hugo_symbols : list of str
        all hugo symbols present in given data
    variant_classifications : list of str
        all variant classifications present in given data
    sorted(cluster_assignments) : list of int
        all cluster assignments present in given data, in order
    participant_maf: pd.DataFrame
        maf_df of participant specific data after being filtered by hugo, variant, and cluster
    final_sample_maf
        maf_df of sample specific data after being filted by hugo, variant, and cluster

    """
    start_pos = 'Start_position' or 'Start_Position'
    end_pos = 'End_position' or 'End_Position'
    protein_change = 'Protein_change' or 'Protein_Change'
    t_ref_count = 't_ref_count' or 't_ref_count_post_forcecall'
    t_alt_count = 't_alt_count' or 't_alt_count_post_forcecall'
    n_ref_count = 'n_ref_count' or 'n_ref_count_post_forcecall'
    n_alt_count = 'n_alt_count' or 'n_alt_count_post_forcecall'

    default_maf_cols = [
        'Hugo_Symbol',
        'Chromosome',
        start_pos,
        end_pos,
        protein_change,
        'Variant_Classification',
        t_ref_count,
        t_alt_count,
        n_ref_count,
        n_alt_count
    ]

    maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    start_pos_id = maf_df.columns[maf_df.columns.isin(['Start_position', 'Start_Position'])][0]
    alt_allele_id = maf_df.columns[maf_df.columns.isin(['Tumor_Seq_Allele2', 'Tumor_Seq_Allele'])][0]
    sample_id_col = maf_df.columns[maf_df.columns.isin(['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id', 'Sample_id'])][0]
    maf_df['Sample_ID'] = maf_df[sample_id_col]
    maf_df['id'] = maf_df.apply(lambda x: get_unique_identifier(x, start_pos=start_pos_id, alt=alt_allele_id), axis=1)
    maf_df.set_index('id', inplace=True, drop=True)

    maf_cols_options = maf_df.dropna(axis=1, how='all').columns.tolist()
    maf_cols_value = list((set(default_maf_cols) | set(cols)) & set(maf_cols_options))
    hugo_symbols = maf_df['Hugo_Symbol'].unique()
    variant_classifications = maf_df['Variant_Classification'].unique()
    cluster_assignments = [] if 'Cluster_Assignment' not in maf_df else maf_df['Cluster_Assignment'].unique()

    filtered_maf_df = maf_df.copy()
    if hugo:
        filtered_maf_df = filtered_maf_df[filtered_maf_df['Hugo_Symbol'].isin(hugo)]
    if variant:
        filtered_maf_df = filtered_maf_df[filtered_maf_df['Variant_Classification'].isin(variant)]
    if cluster and 'Cluster_Assignment' in maf_df:
        filtered_maf_df = filtered_maf_df[filtered_maf_df['Cluster_Assignment'].isin(cluster)]

    sample_options = filtered_maf_df['Sample_ID'].unique()

    filtered_maf_df = filtered_maf_df.sort_values(['Chromosome', start_pos_id]).dropna(axis=1, how='all')

    # todo put this back (to link participant and sample tables)
    # if selected_rows:
    #     maf_df_copy = maf_df_copy.loc[selected_rows]
    # elif filtered_rows:
    #     maf_df_copy = maf_df_copy.loc[filtered_rows]

    # pull all columns that differ between samples
    # use <= so we don't accidentally catch columns that have all NaNs for certain mutations (nunique == 0)
    columns_equivalent = filtered_maf_df.groupby('id').nunique().le(1).all()
    sample_cols = columns_equivalent[~columns_equivalent].index.tolist()

    # generate participant-level (cols w/ no difference between samples) and sample-level mafs
    participant_maf = filtered_maf_df[~filtered_maf_df.index.duplicated(keep='first')].drop(columns=sample_cols)
    # todo change to sort samples by timing
    sample_maf = filtered_maf_df[sample_cols].reset_index().sort_values(['id', 'Sample_ID']).set_index(['id', 'Sample_ID']).unstack(1)

    return [
        maf_cols_options,
        maf_cols_value,
        hugo_symbols,
        variant_classifications,
        sorted(cluster_assignments),
        participant_maf,
        sample_maf
    ]

def maf_callback_return(maf_cols_options, values, maf_cols_value, participant_maf, table_size, custom_colors, hugo_symbols, variant_classifications, cluster_assignments, final_sample_maf):
    """Mutation Table callback functions return (accociated with mutation table component outputs)"""
    return [
        maf_cols_options,
        values,
        dash_table.DataTable(
            id='mutation-table',
            data=participant_maf.to_dict('records'),
            columns=[{'name': i, 'id': i, 'selectable': True} for i in values if i in list(participant_maf)],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            row_selectable='multi',
            column_selectable='multi',  # todo add fixed_columns dict (to freeze columns)
            page_action='native',
            page_current=0,
            page_size=table_size,
            style_data_conditional=gen_style_data_conditional(participant_maf, custom_colors, maf_cols_value)
        ),
        hugo_symbols,
        variant_classifications,
        cluster_assignments,
        dash_table.DataTable(
            id='mutation-sample-table',
            columns=[{'name': [col, s_id], 'id': f'{col}_{s_id}'} for col, s_id in final_sample_maf.columns if col in values],
            data=[
                {
                    **{'': final_sample_maf.index[n]},
                    **{f'{col}_{s_id}': y for (col, s_id), y in data},
                }
                for (n, data) in [
                    *enumerate([list(x.items()) for x in final_sample_maf.T.to_dict().values()])
                ]
            ],
            merge_duplicate_headers=True,
            page_action='none',  # disables paging, renders all data at once
            page_size=table_size
        )
    ]

def gen_maf_table(data: PatientSampleData, idx, cols, hugo, table_size, variant, cluster, custom_colors):
    """Mutation table callback function with parameters being the callback inputs and returns being callback outputs."""
    df = data.participant_df
    maf_cols_options, maf_cols_value, hugo_symbols, variant_classifications, cluster_assignments, filtered_maf_df, final_sample_maf = gen_maf_columns(df, idx, cols, hugo, variant, cluster)

    return maf_callback_return(maf_cols_options, maf_cols_value, maf_cols_value, filtered_maf_df, table_size, custom_colors, hugo_symbols, variant_classifications, cluster_assignments, final_sample_maf)

def internal_gen_maf_table(data: PatientSampleData, idx, cols, hugo, table_size, variant, cluster, custom_colors):
    """Mutation table internal callback function with parameters being the callback inputs and returns being callback outputs."""
    df = data.participant_df
    maf_cols_options, maf_cols_value, hugo_symbols, variant_classifications, cluster_assignments, filtered_maf_df, final_sample_maf = gen_maf_columns(df, idx, cols, hugo, variant, cluster)

    return maf_callback_return(maf_cols_options, cols, maf_cols_value, filtered_maf_df, table_size, custom_colors, hugo_symbols, variant_classifications, cluster_assignments, final_sample_maf)
