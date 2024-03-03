"""MutationTableComponent.py module

Interactive Mutation Table with column selection, sorting, selecting, and filtering

"""
import os.path

import pandas as pd
import numpy as np
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from functools import lru_cache

from AnnoMate.ReviewDataApp import AppComponent
from AnnoMate.AppComponents.utils import cluster_color, get_unique_identifier
from AnnoMate.DataTypes.PatientSampleData import PatientSampleData


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
            Input('mutation-table', 'page_current'),
            Input('mutation-table', 'sort_by'),
            Input('mutation-table', 'filter_query'),
            Input('mutation-table', 'derived_viewport_selected_row_ids')
        ],
        callback_output=[
            Output('column-selection-dropdown', 'options'),
            Output('column-selection-dropdown', 'value'),
            Output('hugo-dropdown', 'options'),
            Output('variant-classification-dropdown', 'options'),
            Output('cluster-assignment-dropdown', 'options'),
            Output('mutation-table', 'data'),
            Output('mutation-table', 'page_size'),
            Output('mutation-table', 'columns'),
            Output('mutation-table', 'style_data_conditional'),
            Output('mutation-table', 'selected_row_ids'),
            Output('mutation-table', 'selected_rows'),
            Output('mutation-sample-table', 'data'),
            Output('mutation-sample-table', 'page_size'),
            Output('mutation-sample-table', 'columns'),
            Output('mutation-sample-table', 'style_data_conditional'),
            Output('mutation-selected-ids', 'value'),
            Output('mutation-selected-ids', 'options'),
            Output('mutation-filtered-ids', 'value'),
            Output('mutation-filtered-ids', 'options')
        ],
        callback_state=[
            State('mutation-selected-ids', 'value'),
            State('mutation-table', 'derived_viewport_row_ids')
        ],
        new_data_callback=update_mutation_tables,
        internal_callback=update_mutation_tables
    )


DEFAULT_PAGE_SIZE = 10


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
                        options=[5,10,20,30],
                        value=DEFAULT_PAGE_SIZE
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
            ]),
        ),

        html.Div([
            dbc.Row([
                dbc.Col([
                    #html.Div(
                    dash_table.DataTable(
                        id='mutation-table',
                        # columns=default_columns,  # todo add fixed_columns dict (to freeze columns)
                        row_selectable='multi',

                        page_current=0,
                        page_size=DEFAULT_PAGE_SIZE,
                        page_action='custom',

                        filter_action='custom',
                        filter_query='',

                        sort_action='custom',
                        sort_mode='multi',
                        sort_by=[]
                    #    ), id='mutation-table-component'
                    )
                ], width=7),
                dbc.Col([
                    #html.Div(
                    dash_table.DataTable(
                        id='mutation-sample-table',
                        # columns=default_sample_columns,
                        merge_duplicate_headers=True,

                        page_action='none',  # disables paging, renders all data at once
                        page_size=DEFAULT_PAGE_SIZE
                    #    ), id='mutation-sample-table-component'
                    )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P('Note: When performing numeric filtering directly in the table, symbols =,>,>=,<,<= must be used. For filtering multiple inputs at once, use filtering dropdowns above the table.'),
                ]),
            ])
        ]),
        html.Div([
            dcc.Dropdown(
                id='mutation-selected-ids',
                options=[],
                multi=True,
                placeholder='Selected Mutations',
                style={'display': 'none'}
            )
        ]),
        html.Div([
            dcc.Dropdown(
                id='mutation-filtered-ids',
                options=[],
                multi=True,
                placeholder='Remaining Mutations',
                style={'display': 'none'}
            )
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
        text box background color

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
    cluster_assignment = maf_df.columns[maf_df.columns.isin(['Cluster_Assignment', 'cluster'])][0]

    if cluster_assignment in maf_cols_value:
        for n in maf_df[cluster_assignment].unique():
            style_data_conditional.append(format_style_data(cluster_assignment, n, color=cluster_color(n)))

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


@lru_cache(maxsize=32)
def load_file(filename):
    if os.path.splitext(filename)[1] == '.pkl':
        maf_df = pd.read_pickle(filename)
    else:
        maf_df = pd.read_csv(filename, sep='\t') # prev had sep \t

    start_pos_id = maf_df.columns[maf_df.columns.isin(['Start_position', 'Start_Position'])][0]
    alt_allele_id = maf_df.columns[maf_df.columns.isin(['Tumor_Seq_Allele2', 'Tumor_Seq_Allele'])][0]
    sample_id_col = \
    maf_df.columns[maf_df.columns.isin(['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id', 'Sample_id'])][0]
    maf_df['Sample_ID'] = maf_df[sample_id_col]
    maf_df['id'] = maf_df.apply(lambda x: get_unique_identifier(x, start_pos=start_pos_id, alt=alt_allele_id), axis=1)
    maf_df.set_index('id', inplace=True, drop=True)

    maf_cols_options = maf_df.dropna(axis=1, how='all').columns.tolist()

    # pull all columns that differ between samples
    # use <= so we don't accidentally catch columns that have all NaNs for certain mutations (nunique == 0)
    columns_equivalent = maf_df.groupby('id', sort=False).nunique().le(1).all()

    return maf_df, maf_cols_options, columns_equivalent

#def update_mutation_tables(data: PatientSampleData, idx, cols, hugo, table_size, variant, cluster, page_current, sort_by, filter_query, viewport_selected_row_ids, prev_selected_ids, viewport_ids, custom_colors=None, default_maf_participant_cols=None, default_maf_sample_cols=None, maf_hugo_col=None, maf_variant_class_col=None, maf_cluster_col=None, maf_sample_id_col=None):
def update_mutation_tables(
        data: PatientSampleData, 
        idx, cols, hugo, table_size, variant, cluster, page_current, sort_by, filter_query, viewport_selected_row_ids, prev_selected_ids, viewport_ids, 
        custom_colors=None, default_maf_sample_cols=None, maf_hugo_col=None, maf_chromosome_col=None, maf_start_pos_col=None, maf_end_pos_col=None, 
        maf_protein_change_col=None, maf_variant_class_col=None, maf_cluster_col=None, maf_sample_id_col=None
    ):
    """Generate mutation table columns from selected columns and filtering dropdowns.

    Parameters
    ----------
    data
        PatientSampleData containing participant and sample dfs
    idx
        Index - current participant
    cols
        column selection dropdown value
    hugo
        hugo symbol filtering dropdown value
    table_size : int
        number of mutations displayed in
    variant
        variant classification filtering dropdown value
    cluster
        cluster assignment filtering dropdown value
    page_current
        current page from mutation table arrows 
    sort_by
        table sorting input 
    filter_query
        table filtering input (default tabel filtering, not dropdowns)
    viewport_selected_row_ids
        selected rows in the mutation table
    prev_selected_ids
    viewport_ids
    custom_colors : list of lists
        kwarg - Specify colummn colors in mutation table with format:
        [[column_id_1, filter_query_1, text_color_1, background_color_1]]
    default_maf_sample_cols
        kwarg - List of default columns to be displayed on the sample side of the mutation table 
    maf_hugo_col
        kwarg - Name of the hugo symbol column in the maf file 
    maf_chromosome_col
        kwarg - Name of the chromosome column in the maf file 
    maf_start_pos_col
        kwarg - Name of the start position column in the maf file 
    maf_end_pos_col
        kwarg - Name of the end position column in the maf file 
    maf_protein_change_col
        kwarg - Name of the protein change column in the maf file 
    maf_variant_class_col
        kwarg - Name of the variant classification column in the maf file 
    maf_cluster_col
        kwarg - Name of the cluster assignment column in the maf file 
    maf_sample_id_col
        kwarg - Name of the sample id column in the maf file 

    Returns
    -------
    Corresponding values to callback outputs (in corresponding order)
    
    """
    df = data.participant_df
    default_maf_participant_cols=[
        maf_hugo_col,
        maf_chromosome_col,
        maf_start_pos_col,
        maf_end_pos_col,
        maf_protein_change_col,
        maf_variant_class_col
    ]
    default_maf_cols = default_maf_participant_cols.copy()
    default_maf_cols.extend(default_maf_sample_cols)

    #####
    # load maf from file
    if 'maf_df_pickle' in df:
        maf_df, maf_cols_options, columns_equivalent = load_file(df.loc[idx, 'maf_df_pickle'])
    else:
        maf_df, maf_cols_options, columns_equivalent = load_file(df.loc[idx, 'maf_fn'])

    # get the columns displayed in the table, ensuring these columns are present in the maf 
    if not cols:  # Nothing selected for columns
        maf_cols_value = [c for c in default_maf_cols if c in maf_cols_options]# list(set(default_maf_cols) & set(maf_cols_options))
    else:
        maf_cols_value = [c for c in cols if c in maf_cols_options] #list(set(cols) & set(maf_cols_options))

    # options and values for filtering dropdowns 
    # if none is input for the column name, the dropdown list will be empty 
    hugo_symbols = maf_df[maf_hugo_col].unique() if maf_hugo_col else []
    hugo_value_in_maf = list(set(hugo) & set(hugo_symbols)) if hugo else None
    variant_classifications = maf_df[maf_variant_class_col].unique() if maf_variant_class_col else []
    variant_in_maf = list(set(variant) & set(variant_classifications)) if variant else None
    cluster_assignments = maf_df[maf_cluster_col].unique() if maf_cluster_col else []
    cluster_in_maf = list(set(cluster) & set(cluster_assignments)) if cluster else None
    sample_options = maf_df[maf_sample_id_col].unique()

    # apply the filter dropdown values to the maf
    filtered_maf_df = maf_df.copy()
    if hugo_value_in_maf:
        filtered_maf_df = filtered_maf_df[filtered_maf_df[maf_hugo_col].isin(hugo_value_in_maf)]
    if variant_in_maf:
        filtered_maf_df = filtered_maf_df[filtered_maf_df[maf_variant_class_col].isin(variant_in_maf)]
    if cluster_in_maf:
        filtered_maf_df = filtered_maf_df[filtered_maf_df[maf_cluster_col].isin(cluster_in_maf)]

    # built in table filtering 
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            filtered_maf_df = filtered_maf_df.loc[getattr(filtered_maf_df[col_name], operator)(filter_value)]
        elif operator == 'contains':
            filtered_maf_df = filtered_maf_df.dropna(axis=0, subset=[col_name])
            filtered_maf_df = filtered_maf_df.loc[filtered_maf_df[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            filtered_maf_df = filtered_maf_df.loc[filtered_maf_df[col_name].str.startswith(filter_value)]

    if len(sort_by):
        filtered_maf_df = filtered_maf_df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    # if filtered dataframe is not empty, remove columns with only NaNs or Nones 
    filtered_maf_empty = filtered_maf_df.shape[0] == 0
    if not filtered_maf_empty:
        filtered_maf_df.replace(['None', None], np.nan, inplace=True)
        maf_cols_value = [c for c in maf_cols_value if c in list(filtered_maf_df.dropna(axis=1, how='all'))] 
        filtered_maf_df = filtered_maf_df.dropna(axis=1, how='all')
        

    # generate sample-level dataframe (cols that differ between samples)
    if filtered_maf_empty:
        sample_cols = default_maf_sample_cols.copy()
        sample_cols.append(maf_sample_id_col)
        sample_maf = pd.DataFrame(columns=sample_cols)
        sample_maf.loc[:, maf_sample_id_col] = sample_options
        sample_maf.index.name = 'id'
    else:
        sample_cols = columns_equivalent[~columns_equivalent].index.tolist()
        sample_maf = filtered_maf_df[sample_cols]

    # todo change to sort samples by timing
    sample_maf = sample_maf.reset_index().sort_values(['id', maf_sample_id_col]).set_index(
        ['id', maf_sample_id_col]).unstack(1)
    sample_maf['id'] = sample_maf.index.tolist()

    # generate participant-level (cols w/ no difference between samples)
    participant_maf = filtered_maf_df[~filtered_maf_df.index.duplicated(keep='first')]
    if sample_cols:
        participant_maf.drop(columns=sample_cols, inplace=True)
    participant_maf['id'] = participant_maf.index.tolist()

    #####

    participant_table_data = participant_maf.iloc[page_current * table_size: (page_current + 1) * table_size]  #.to_dict('records')
    participant_columns = [{'name': i, 'id': i, 'selectable': True} for i in maf_cols_value if i in list(participant_maf)]

    sample_table = sample_maf.loc[participant_table_data.index]
    sample_table_data = [{
        **{'': sample_table.index[n]},
        **{f'{col}_{s_id}': y for (col, s_id), y in data},
    }
        for (n, data) in [
            *enumerate([list(x.items()) for x in sample_table.T.to_dict().values()])
        ]
    ]
    sample_columns_s_id = [{'name': [col, s_id], 'id': f'{col}_{s_id}'} for col, s_id in sample_table.columns if col in maf_cols_value]

    prev_selected_ids = [] if prev_selected_ids is None else prev_selected_ids

    # There is a bug where the selected_rows attribute is erased after certain table actions (filter query, table size)
    # To allow for keeping the selected rows after these callback triggers,
    # check that only the selected rows have changed (not the data).
    # The table size change is more tricky, but checking that button selections only go down to zero from size one.
    #   There are some cases where this is unintended (when only 1 row is selected), but it's the best I can do for now.
    # Follow bug reports here: https://github.com/plotly/dash-table/issues/924,
    #                          https://github.com/plotly/dash-table/issues/938
    if 'mutation-table.derived_viewport_selected_row_ids' in ctx.triggered_prop_ids.keys() and set(viewport_ids) == set(participant_table_data.index):
        updated_selected_ids = list((set(prev_selected_ids) - set(viewport_ids)) | set(viewport_selected_row_ids))
        if len(updated_selected_ids) == 0 and len(prev_selected_ids) > 1:
            updated_selected_ids = prev_selected_ids
    else:
        updated_selected_ids = prev_selected_ids

    if updated_selected_ids and len(updated_selected_ids):
        derived_viewport_selected_row_ids = participant_table_data.loc[set(updated_selected_ids) & set(participant_table_data.index)].index
        derived_viewport_selected_rows = [participant_table_data.index.get_loc(vis_row_id) for vis_row_id in derived_viewport_selected_row_ids]
    else:
        derived_viewport_selected_row_ids = []
        derived_viewport_selected_rows = []

    return [
        maf_cols_options,
        maf_cols_value,
        hugo_symbols,
        variant_classifications,
        sorted(cluster_assignments),
        participant_table_data.to_dict('records'),  # participant_data
        table_size,
        participant_columns,
        gen_style_data_conditional(participant_maf, custom_colors, maf_cols_value),
        derived_viewport_selected_row_ids,
        derived_viewport_selected_rows,
        sample_table_data,
        table_size,
        sample_columns_s_id,
        gen_style_data_conditional(participant_maf, custom_colors, maf_cols_value),
        updated_selected_ids,
        maf_df.index.unique().tolist(),
        participant_maf.index.tolist(),
        maf_df.index.unique().tolist(),
    ]


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    """Split filter query into operator and value

    Taken from dash documentation at https://dash.plotly.com/datatable/callbacks on the Python-Driven Filtering, Paging, Sorting Page.
    """
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3
