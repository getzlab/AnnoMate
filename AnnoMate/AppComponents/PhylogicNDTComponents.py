"""PhylogicNDTComponents.py module

PhylogicNDT CCF Plot and Trees implemented in the PatientReviewer and PhylogicNDTReviewer

PhylogicNDT PMF Plot implemented in the PhylogicNDTReviewer

"""

import pandas as pd
import numpy as np
from dash import dcc
from dash import html, ctx
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_cytoscape as cyto
import re
import scipy.stats as ss

from AnnoMate.ReviewDataApp import AppComponent
from AnnoMate.AppComponents.utils import cluster_color, get_unique_identifier
from AnnoMate.DataTypes.PatientSampleData import PatientSampleData


# --------------------- PhylogicNDT CCF Plot and Tree ------------------------

def gen_phylogicNDT_app_component():
    """Generate PhylogicNDT CCF Plot and Tree components"""
    return AppComponent(
        'PhylogicNDT Results',
        layout=gen_phylogicNDT_components_layout(),

        callback_input=[
            Input('time-scale-checklist', 'value'),
            Input('tree-dropdown', 'value'),
            Input('save-tree-button', 'n_clicks')
        ],
        callback_output=[
            Output('ccf-plot', 'figure'),
            Output('tree-dropdown', 'options'),
            Output('tree-dropdown', 'value'),
            Output('phylogicNDT-tree-component', 'children'),
            Output('phylogicNDT-tree', 'generateImage')
        ],
        new_data_callback=gen_phylogicNDT_graphics,
        internal_callback=internal_gen_phylogicNDT_graphics
    )

def gen_phylogicNDT_components_layout():
    """Generate phylogicNDT CCF Plot and Tree layout"""
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
                            id='phylogicNDT-tree',
                            elements=[],
                            style={'width': '100%', 'height': '450px'},
                        ),
                        id='phylogicNDT-tree-component'
                    ),
                    dcc.Dropdown(
                        id='tree-dropdown',
                        options=[]
                    ),
                    html.Button('save', id='save-tree-button'),
                ], width=4, align='center')
            ])
        ])
    ])

def gen_ccf_plot(df, idx, time_scaled, samples_df, maf_participant_id_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col, maf_cluster_col):
    """Generate CCF plot including treatment bars.

    Parameters
    ----------
    df
        participant level DataFrame
    idx
    time_scaled
        time scaled checkbox value
    samples_df
        Samples dataframe, containing collection date data as 'collection_date_dfd' data

    Returns
    -------
    ccf_plot : make_subplots()

    """
    # todo add more categories
    treatment_category_colors = {
        'Chemotherapy': 'MidnightBlue',
        'Hormone/Endocrine therapy': 'MistyRose',
        'Precision/Targeted therapy': 'Plum',
        'Immunotherapy': 'Orange'
    }

    cluster_df = pd.read_csv(df.loc[idx, 'cluster_ccfs_fn'], sep='\t', usecols=['Cluster_ID', 'Sample_ID',
                                                                                'postDP_ccf_mean', 'postDP_ccf_CI_low',
                                                                                'postDP_ccf_CI_high'])
    samples_list = cluster_df['Sample_ID'].unique()

    # todo replace this with using sif file - to ensure all collection dates are present and correct
    # pull collection dates from sample table, robust to missing values
    timing_data = {sample: samples_df.loc[sample, 'collection_date_dfd'] if sample in samples_df.index else 0 for sample in samples_list}
    samples_in_order = sorted(timing_data.keys(), key=lambda k: int(timing_data[k]))
    ordered_samples_dict = {s: o for s, o in zip(samples_in_order, np.arange(len(samples_in_order)))}

    # apply dates and sample order to cluster df
    cluster_df.loc[:, 'dfd'] = cluster_df['Sample_ID'].apply(lambda s: int(timing_data[s]))
    cluster_df.loc[:, 'order'] = cluster_df['Sample_ID'].apply(lambda s: ordered_samples_dict[s])

    if 'Time Scaled' in time_scaled:
        scatter_x = 'dfd'
        rect_x = 5
    else:
        scatter_x = 'order'
        rect_x = 6

    if 'treatments_fn' in df:
        treatments_df = pd.read_csv(df.loc[idx, 'treatments_fn'], sep='\t', comment='#')
        treatments_in_frame_df = treatments_df[(treatments_df['stop_date_dfd'] >= int(timing_data[samples_in_order[0]])) &
                                           (treatments_df['start_date_dfd'] <= int(timing_data[samples_in_order[-1]]))]

    # get mutation counts
    mut_ccfs = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    # participant_id = mut_ccfs.columns[mut_ccfs.columns.isin(['Patient_ID', 'Participant_ID', 'participant_id'])][0]
    # start_pos_id = mut_ccfs.columns[mut_ccfs.columns.isin(['Start_position', 'Start_Position'])][0]
    # cluster = mut_ccfs.columns[mut_ccfs.columns.isin(['Cluster_Assignment', 'cluster'])][0]
    mut_count_dict = mut_ccfs.drop_duplicates([
        maf_participant_id_col,
        maf_hugo_col,
        maf_chromosome_col,
        maf_start_pos_col,
        maf_cluster_col
    ]).groupby(maf_cluster_col).count()[maf_participant_id_col].to_dict()

    cluster_colors = [cluster_color(i) for i in cluster_df['Cluster_ID'].unique()]
    cluster_df['Cluster_ID'] = cluster_df['Cluster_ID'].astype(str)

    ccf_plot = make_subplots(rows=2, cols=1, row_heights=[15,1], shared_xaxes=True)

    for c, color in zip(cluster_df['Cluster_ID'].unique(), cluster_colors):
        this_cluster = cluster_df[cluster_df['Cluster_ID'] == c]
        for i in np.arange(this_cluster.shape[0] - 1):
            x = [this_cluster.iloc[i, rect_x], this_cluster.iloc[i + 1, rect_x], this_cluster.iloc[i + 1, rect_x],
                 this_cluster.iloc[i, rect_x], this_cluster.iloc[i, rect_x]]
            y = [this_cluster.iloc[i, 4], this_cluster.iloc[i + 1, 4], this_cluster.iloc[i + 1, 3],
                 this_cluster.iloc[i, 3], this_cluster.iloc[i, 4]]
            # plot points
            legend = False
            if i == 0:
                legend = True

            ccf_plot.add_trace(
                go.Scatter(
                    x=this_cluster[scatter_x],
                    y=this_cluster['postDP_ccf_mean'],
                    legendgroup=f'group{c}',
                    name=c,
                    marker_color=color,
                    mode='markers',
                    showlegend=legend
                ),
                row=1, col=1
            )
            #confidence interval
            ccf_plot.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    legendgroup=f'group{c}',
                    name=f'{c}',
                    fill="toself",
                    fillcolor=color,
                    line_color=color,
                    opacity=0.4,
                    mode='none',
                    showlegend=False
                ),
                row=1, col=1
            )
            # line
            ccf_plot.add_trace(
                go.Scatter(
                    x=[this_cluster.iloc[i, rect_x], this_cluster.iloc[i + 1, rect_x]],
                    y=[this_cluster.iloc[i, 2], this_cluster.iloc[i + 1, 2]],
                    legendgroup=f'group{c}',
                    name=f'{c}',
                    line_width=min(mut_count_dict[int(c)], 15),
                    line_color=color,
                    opacity=0.4,
                    showlegend=False
                ),
                row=1, col=1
            )

    ccf_plot.update_traces(marker_size=15)
    ccf_plot.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    ccf_plot.update_layout(legend={'traceorder': 'reversed'})
    ccf_plot.update_yaxes(title='ccf(x)', dtick=0.1, ticks='outside', showline=True, linecolor='black', range=[-0.03,1.05], showgrid=False)
    ccf_plot.update_xaxes(ticks='outside', showline=True, linecolor='black', showgrid=False)
    if 'Time Scaled' in time_scaled:
        ccf_plot.update_xaxes(title='Time (dfd)')
    else:
        ccf_plot.update_xaxes(title='Samples (timing - dfd)', tickvals=np.arange(len(samples_in_order)),
                         ticktext=[f'{s} ({timing_data[s]})' for s in samples_in_order])
    ccf_plot.data = ccf_plot.data[::-1]  # make the circles appear on top layer

    ccf_plot.add_trace(
        go.Scatter(
            x=this_cluster[scatter_x],
            y=[0,0],
            line_width=20,
            line_color='white',
            fill='toself',
            showlegend=False
        ),
        row=2, col=1
    )

    if 'Time Scaled' in time_scaled and 'treatments_fn' in df:
        for start, stop, drug, drug_combo, category, stop_reason, post_status in zip(treatments_in_frame_df.start_date_dfd,
                                                                                     treatments_in_frame_df.stop_date_dfd,
                                                                                     treatments_in_frame_df.drugs,
                                                                                     treatments_in_frame_df.drug_combination,
                                                                                     treatments_df.categories,
                                                                                     treatments_in_frame_df.stop_reason,
                                                                                     treatments_in_frame_df.post_status):
            drug = drug_combo if pd.isna(drug) else drug

            # todo deal with overlapping treatments
            ccf_plot.add_trace(
                go.Scatter(
                    # todo implement 'order' for x
                    x=[max(start, int(timing_data[samples_in_order[0]])), min(stop, int(timing_data[samples_in_order[-1]]))],
                    y=[0,0],
                    line_width=20,
                    line_color=treatment_category_colors[category] if category in treatment_category_colors.keys() else 'gray',
                    fill='toself',
                    hovertemplate = '<extra></extra>' +
                        f'Treatment Regimen: {drug} <br>' +
                        f'Stop Reason: {stop_reason} <br>' +
                        f'Post Status: {post_status}',
                    showlegend=False
                ),
                row=2, col=1
            )
            ccf_plot.add_vline(
                x=max(start, int(timing_data[samples_in_order[0]])),
                line_width=2,
                line_color='black',
                row=2, col=1
            )
            ccf_plot.add_vline(
                    x=min(stop, int(timing_data[samples_in_order[-1]])),
                    line_width=2,
                    line_color='black',
                    row=2, col=1
            )

    ccf_plot.update_yaxes(row=2, visible=False)
    ccf_plot.update_xaxes(row=1, visible=False, showticklabels=False)

    return ccf_plot

def gen_stylesheet(cluster_list):
    """Format PhylogicNDT tree to have correct cluster colors and labels

    Parameters
    ----------
    cluster_list
        list of clusters in given data

    Returns
    -------
    stylesheet : list of dicts

    """
    stylesheet = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'width': '50%',
                'height': '50%',
                'text-halign':'center',
                'text-valign':'center',
                'color': 'white'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'label': 'data(label)',
                'text-halign':'center',
                'text-valign':'center',
                'color': 'black',
                'text-wrap': 'wrap',
                'font-weight': 'bold'
            }
        }
    ]

    for node in cluster_list:
        stylesheet.append({
            'selector': ('node[label = "%s"]' % node),
            'style': {
                'background-color': cluster_color(int(node))
            }
        })
        stylesheet.append({
            'selector': ('edge[target = "%s"]' % f'cluster_{node}'),
            'style': {
                'line-color': cluster_color(int(node))
            }
        })

    return stylesheet

def gen_driver_edge_labels(drivers, cluster_hugo_list):
    """Add driver mutation label to edges of clusters containing that driver

    Parameters
    ----------
    drivers: pd.DataFrame()
        DataFrame from the drivers kwarg file
    cluster_hugo_list: list of str
        list of hugo symbols associated with a particular cluster

    Returns
    -------
    label: str

    """
    label = ''
    for driver in drivers.drivers:
        if driver in cluster_hugo_list:
            label += ('%s \n' % driver)

    return label

def gen_phylogicNDT_tree(df, idx, tree_num, drivers_fn, maf_start_pos_col, maf_cluster_col, maf_hugo_col, tree_meta_col='n_iter'):
    """Generate PhlogicNDT tree and dropdown to choose from all possible trees.

    Parameters
    ----------
    df
        participant level DataFrame
    idx
    tree_num
        number assigned to the chosen tree that is to be displayed
    drivers_fn
        name of the drivers file passed into to gen_review_app as kwarg
    tree_meta_col
        Which column from the tree file to display along with the tree name in the dropdown. See data in 'build_tree_posterior_fn'

    Returns
    -------
    cyto.Cytoscape
        the tree figure
    possible_trees : list of str
        possible tree options for dropdown

    """
    tree_df = pd.read_csv(df.loc[idx, 'build_tree_posterior_fn'], sep='\t')
    maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    maf_df.drop_duplicates(subset=maf_start_pos_col, inplace=True)
    if drivers_fn:
        drivers = pd.read_csv(drivers_fn, header=None, names=['drivers'])

    # cluster_assignment_col = maf_df.columns[maf_df.columns.isin(['Cluster_Assignment', 'cluster'])][0]
    cluster_assignments = maf_df[maf_cluster_col].unique().tolist()
    possible_trees = []
    possible_trees_edges = []
    clusters = {}
    cluster_count = {}
    cluster_list = []

    trees = tree_df.loc[:, 'edges']
    for i, tree in enumerate(trees):
        possible_trees_edges.append(tree.split(','))
        tree_label = f'Tree {i+1}'
        if tree_meta_col:
            tree_label = tree_label + f' ({tree_meta_col}={tree_df[tree_meta_col][i]})'
        possible_trees.append(tree_label)

    for i in range(len(cluster_assignments)):
        clusters[cluster_assignments[i]] = [hugo for clust, hugo in zip(maf_df[maf_cluster_col], maf_df[maf_hugo_col]) if clust == cluster_assignments[i]]

    for clust in clusters:
        cluster_count[clust] = len(clusters[clust])

    edges = possible_trees_edges[tree_num]

    for i in edges:
        new_list = i.split('-')
        for j in new_list:
            if (j !='None') & (j not in cluster_list):
                cluster_list.append(j)
    cluster_list = sorted(cluster_list)

    nodes = [{'data': {'id': 'normal', 'label': 'normal'}, 'position': {'x': 0, 'y': 0}}]

    nodes.extend([
        {
            'data': {'id': f'cluster_{cluster}', 'label': cluster},
            'position': {'x': 50 * int(cluster), 'y': -50 * int(cluster)}
        }
        for cluster in cluster_list
    ])

    edges_list = []
    nodes_copy = nodes.copy()
    for edge in edges:
        nodes_copy = edge.split('-')
        if nodes_copy[0]!='None':
            nodes_copy = list(map(int,edge.split('-')))
            edges_list.append(nodes_copy)

    if drivers_fn:
        edges = [{'data': {'source': 'normal', 'target': 'cluster_1', 'label': f'{cluster_count[1]}\n{gen_driver_edge_labels(drivers, clusters[1])}'}}]
        edges.extend([
            {'data': {'source': f'cluster_{edge[0]}', 'target': f'cluster_{edge[1]}', 'label': f'{cluster_count[edge[1]]}\n{gen_driver_edge_labels(drivers, clusters[edge[1]])}'}}
            for edge in edges_list
        ])
    else:
        edges = [{'data': {'source': 'normal', 'target': 'cluster_1', 'label': str(cluster_count[1])}}]
        edges.extend([
            {'data': {'source': f'cluster_{edge[0]}', 'target': f'cluster_{edge[1]}', 'label': str(cluster_count[edge[1]])}}
            for edge in edges_list
        ])

    elements = nodes + edges

    stylesheet = gen_stylesheet(cluster_list)

    return [
        cyto.Cytoscape(
            id='phylogicNDT-tree',
            style={'width': '100%', 'height': '450px'},
            layout={
                'name': 'breadthfirst',
                'roots': '[id="normal"]'
            },
            elements=elements,
            stylesheet=stylesheet,
            userZoomingEnabled=False
        ),
        possible_trees
    ]

def gen_phylogicNDT_graphics(
        data: PatientSampleData, 
        idx, time_scaled, chosen_tree, save_tree_button, 
        drivers_fn=None, maf_participant_id_col=None, maf_hugo_col=None, maf_chromosome_col=None, maf_start_pos_col=None, maf_cluster_col=None,
        tree_meta_col='n_iter'
    ):
    """Generate PhylogicNDT CCF plot and trees - new data callback.

    Parameters
    ----------
    data
        PatientSampleData containing participant and sample dfs
    idx
        Index - current participant
    time_scaled
        Time scaled checkbox input (checked or not checked)
    chosen_tree
        Tree chosen in the tree dropdown
    save_tree_button
        Response from the 'save' button under the tree dropdown being clicked
    drivers_fn
        kwarg - File path to csv file of drivers
    maf_participant_id_col
        kwarg - Name of the participant id column in the maf file
    maf_hugo_col
        kwarg - Name of the hugo symbol column in the maf file 
    maf_chromosome_col
        kwarg - Name of the chromosome column in the maf file 
    maf_start_pos_col
        kwarg - Name of the start position column in the maf file 
    maf_cluster_col
        kwarg - Name of the cluster assignment column in the maf file  

    Returns
    -------
    Corresponding values to callback outputs (in corresponding order)

    """
    df = data.participant_df
    samples_df = data.sample_df

    if df.loc[idx, 'cluster_ccfs_fn']:
        ccf_plot = gen_ccf_plot(df, idx, time_scaled, samples_df, maf_participant_id_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col, maf_cluster_col)
        tree, possible_trees = gen_phylogicNDT_tree(df, idx, 0, drivers_fn, maf_start_pos_col, maf_cluster_col, maf_hugo_col, tree_meta_col=tree_meta_col)

        return [ccf_plot, possible_trees, possible_trees[0], tree, dash.no_update]
    else:
        return [go.Figure, [], 0, '', dash.no_update]

def internal_gen_phylogicNDT_graphics(
        data: PatientSampleData, 
        idx, time_scaled, chosen_tree, save_tree_button, 
        drivers_fn=None, maf_participant_id_col=None, maf_hugo_col=None, maf_chromosome_col=None, maf_start_pos_col=None, maf_cluster_col=None,
        tree_meta_col='n_iter'
    ):
    """Generate PhylogicNDT CCF plot and trees - interal callback.
    
    Parameters
    ----------
    data
        PatientSampleData containing participant and sample dfs
    idx
        Index - current participant
    time_scaled
        Time scaled checkbox input (checked or not checked)
    chosen_tree
        Tree chosen in the tree dropdown
    save_tree_button
        Response from the 'save' button under the tree dropdown being clicked
    drivers_fn
        kwarg - File path to csv file of drivers
    maf_participant_id_col
        kwarg - Name of the participant id column in the maf file
    maf_hugo_col
        kwarg - Name of the hugo symbol column in the maf file 
    maf_chromosome_col
        kwarg - Name of the chromosome column in the maf file 
    maf_start_pos_col
        kwarg - Name of the start position column in the maf file 
    maf_cluster_col
        kwarg - Name of the cluster assignment column in the maf file  

    Returns
    -------
    Corresponding values to callback outputs (in corresponding order)

    """
    df = data.participant_df
    samples_df = data.sample_df
    if ctx.triggered:
        if ctx.triggered_id != 'save-tree-button':
            if df.loc[idx, 'cluster_ccfs_fn']:
                tree_num = 0
                for n in chosen_tree.split():
                    if n.isdigit():
                        tree_num = int(n)

                ccf_plot = gen_ccf_plot(df, idx, time_scaled, samples_df, maf_participant_id_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col, maf_cluster_col)
                tree, possible_trees = gen_phylogicNDT_tree(df, idx, tree_num-1, drivers_fn, maf_start_pos_col, maf_cluster_col, maf_hugo_col, tree_meta_col)

                return [ccf_plot, possible_trees, chosen_tree, tree, dash.no_update]
            else:
                return [go.Figure, [], 0, '', dash.no_update]
        else:
            return [dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'type': 'jpg', 'action': 'download'}]


# -------------------------- PhylogicNDT PMF Plot ----------------------------
def gen_ccf_pmf_component():
    return AppComponent(name='CCF pmf Mutation Plot',
                        layout=html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='pmf-plot',
                                              figure=go.Figure())
                                ], width=10),
                                dbc.Col([html.H3('Customize Plot'),
                                         html.H5('Samples:'),
                                         dcc.Checklist(options=[],
                                                       value=[],
                                                       id='sample-selection'),
                                         html.P(''),
                                         html.H5('Group by Cluster:'),
                                         daq.BooleanSwitch(id='group-clusters', on=True),
                                         html.P(''),
                                         html.Button('Submit', id='pmf-button')
                                ], width=2)
                            ])
                        ]),
                        callback_input=[
                            Input('pmf-button', 'n_clicks')
                        ],
                        callback_output=[
                            Output('pmf-plot', 'figure'),
                            Output('sample-selection', 'options'),
                            Output('sample-selection', 'value')
                        ],
                        callback_state=[
                            State('sample-selection', 'value'),
                            State('group-clusters', 'on')
                        ],
                        callback_state_external=[
                            State('mutation-selected-ids', 'value'),  # selected rows
                            State('mutation-filtered-ids', 'value')  # all rows in table after filtering
                        ],
                        new_data_callback=gen_pmf_component,
                        internal_callback=update_pmf_component
                        )


def ccf_pmf_plot(data_df, idx, sample_selection, group_clusters, selected_mut_ids, filtered_mut_ids, maf_sample_id_col, maf_cluster_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col):

    """Plots the CCF pmf distribution for the chosen mutation(s).

    Notes
    -----
    - Displays the pmf distribution as a normalized histogram
    - Samples are shown in separate rows
    - Clusters displayed with different colors, with adjacent bars
    - Given maf file in column 'maf_fn' in the df must be mut_ccfs file

    TODO
    ----
    - Add a star (*) above the mode for each mutation
    - Add an indication of mean?

    """
    mut_ccfs_df = pd.read_csv(data_df.loc[idx, 'maf_fn'], sep='\t')
    mut_ccfs_df['unique_mut_id'] = mut_ccfs_df.apply(get_unique_identifier, axis=1)  # must be mut_ccfs file with default columns
    mut_ccfs_df.set_index('unique_mut_id', inplace=True, drop=False)

    # Use only the selected mutations unless no mutations selected, then use filtered list
    if selected_mut_ids:
        mut_ccfs_df = mut_ccfs_df.loc[selected_mut_ids].copy()
    elif filtered_mut_ids:
        mut_ccfs_df = mut_ccfs_df.loc[filtered_mut_ids].copy()
    # else (if all mutations in table are filtered out and none selected): use all mutations

    sample_list = mut_ccfs_df[maf_sample_id_col].unique()  # todo ensure sorted by collection date
    sample_selection = sample_list if not sample_selection else sample_selection
    mut_ccfs_df = mut_ccfs_df[mut_ccfs_df[maf_sample_id_col].isin(sample_selection)].copy()

    ccfs_headers = [re.search('.*[01].[0-9]+', i) for i in mut_ccfs_df.columns]
    ccfs_headers = [x.group() for x in ccfs_headers if x]
    ccfs_header_dict = {i: re.search('[01].[0-9]+', i).group() for i in ccfs_headers}

    stacked_muts = mut_ccfs_df.set_index([maf_sample_id_col, 'unique_mut_id', maf_cluster_col])[
        ccfs_headers].stack().reset_index().rename(columns={'level_3': 'CCF', 0: 'Probability'}).replace(
        ccfs_header_dict)
    if group_clusters:
        stacked_muts[maf_cluster_col] = stacked_muts[maf_cluster_col].astype(str)
        fig = px.histogram(stacked_muts, x='CCF', y='Probability', facet_row=maf_sample_id_col, barmode='group',
                           height=300 * len(sample_selection), color=maf_cluster_col, histfunc='avg',
                           color_discrete_map=cluster_color(),
                           category_orders={maf_cluster_col: np.arange(1, 100)})
    else:
        fig = px.histogram(stacked_muts, x='CCF', y='Probability', facet_row=maf_sample_id_col, barmode='group',
                           height=300 * len(sample_selection), color='unique_mut_id',
                           labels={'unique_mut_id': 'Mutation'})
        mut_label_dict = {x['unique_mut_id']: f"{x[maf_hugo_col]} - {x[maf_chromosome_col]}:{x[maf_start_pos_col]}" for idx, x
                          in mut_ccfs_df.drop_duplicates('unique_mut_id').iterrows()}
        fig.for_each_trace(lambda t: t.update(name=mut_label_dict[t.name]))

    fig.update_layout(xaxis_tickangle=0, xaxis_ticklabelstep=5)
    fig.update_yaxes(matches=None)

    return fig, sample_list


def gen_pmf_component(data: PatientSampleData, idx, button_clicks, sample_selection,
                      group_clusters, selected_mut_ids, filtered_mut_ids,
                      maf_sample_id_col=None, maf_cluster_col=None, maf_hugo_col=None, 
                      maf_chromosome_col=None, maf_start_pos_col=None):
    # when changing participants, show all mutations at first
    # the filtered and selected mutations input to this function are from the old (previous participant's) MutationTable
    filtered_mut_ids = None
    selected_mut_ids = None

    fig, sample_list = ccf_pmf_plot(data.participant_df, idx, None, group_clusters, selected_mut_ids, filtered_mut_ids, maf_sample_id_col, maf_cluster_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col)

    return [fig, sample_list, sample_list]


def update_pmf_component(data: PatientSampleData, idx, button_clicks, sample_selection,
                         group_clusters, selected_mut_ids, filtered_mut_ids, 
                         maf_sample_id_col=None, maf_cluster_col=None, maf_hugo_col=None, 
                         maf_chromosome_col=None, maf_start_pos_col=None):
    fig, sample_list = ccf_pmf_plot(data.participant_df, idx, sample_selection,
                                    group_clusters, selected_mut_ids, filtered_mut_ids, maf_sample_id_col, maf_cluster_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col)

    return [fig, sample_list, sample_selection]

# ----------------------------- PhylogicNDT Cluster Metrics -------------------------------

def gen_cluster_metrics_component():
    return AppComponent(name='Mutation Types by Cluster',
                        layout=html.Div([
                                    dcc.Graph(id='metric-plot',
                                              figure=go.Figure())
                        ]),
                        callback_output=[
                            Output('metric-plot', 'figure'),
                        ],
                        new_data_callback=gen_cluster_metric_fig,
                        internal_callback=gen_cluster_metric_fig
                        )


def gen_cluster_metric_fig(data: PatientSampleData, idx, maf_variant_type_col=None, maf_variant_class_col=None, maf_cluster_col=None):
    """Generate a figure showing mutation type comparisons across clusters with indication of differences."""
    data_df = data.participant_df

    mut_ccfs_df = pd.read_csv(data_df.loc[idx, 'maf_fn'], sep='\t')
    mut_ccfs_df['unique_mut_id'] = mut_ccfs_df.apply(get_unique_identifier, axis=1)  # mut_ccfs file has default columns
    mut_ccfs_df.drop_duplicates('unique_mut_id', inplace=True)

    # apply functions to specify coding vs. non-coding; silent vs. nonsyn
    mut_ccfs_df['snv_indel'] = mut_ccfs_df[maf_variant_type_col].apply(lambda x: 'SNV' if x == 'SNP' or 'SNV' else 'INDEL')
    mut_ccfs_df['snv_indel'] = pd.Categorical(mut_ccfs_df['snv_indel'], categories=['SNV', 'INDEL'])
    mut_ccfs_df['class'] = mut_ccfs_df[maf_variant_class_col].apply(classify_mut)
    mut_ccfs_df['class'] = pd.Categorical(mut_ccfs_df['class'], categories=['non-coding', 'synonymous', 'non-synonymous'])

    mut_type_counts = mut_ccfs_df.groupby([maf_cluster_col, 'snv_indel']).size().unstack(fill_value=0)
    mut_classes_counts = mut_ccfs_df.groupby([maf_cluster_col, 'class']).size().unstack(fill_value=0)
    mut_classes_counts['coding'] = mut_classes_counts['synonymous'] + mut_classes_counts['non-synonymous']

    mut_counts_df = mut_classes_counts.join(mut_type_counts)
    mut_counts_df_mod = mut_counts_df.copy()
    mut_counts_df_mod.loc['ALL', :] = mut_counts_df_mod.sum()
    mut_counts_df_mod = mut_counts_df_mod.stack().reset_index().rename(columns={'level_1': 'annotation', 0: 'count'})
    mut_counts_df_mod['type'] = mut_counts_df_mod['annotation'].apply(
        lambda x: 'Non/Coding' if 'coding' in x else ('Non/Syn' if 'syn' in x else 'SNV/INDEL'))
    # turn these into pie charts (one for each mut comparison type)
    num_clusters = len(mut_counts_df_mod[maf_cluster_col].unique())
    fig = make_subplots(
        rows=2, cols=num_clusters, 
        specs=[[{"type": "pie"} for _ in range(0, num_clusters)] for _ in range(0, 2)], 
        subplot_titles=[i for i in mut_counts_df_mod[maf_cluster_col].unique()],
        row_titles=[i for i in mut_counts_df_mod['type'].unique()]
    )
    for i, annotations in enumerate(zip(['non-coding', 'synonymous'], ['coding', 'non-synonymous'])):
        for j, clust in enumerate(mut_counts_df_mod[maf_cluster_col].unique()):
            filterby_annotation_df = mut_counts_df_mod[mut_counts_df_mod['annotation'].isin(annotations)]
            filterby_cluster_df = filterby_annotation_df[filterby_annotation_df[maf_cluster_col] == clust]
            filterby_cluster_df.set_index('annotation', inplace=True)

            fig.add_trace(go.Pie(
                values=filterby_cluster_df['count'],
                domain=dict(x=[i/num_clusters, (i+1)/num_clusters]),
                labels=filterby_cluster_df['count'].index,
                legendgroup=i,
                textinfo='value',
            ), row=(i+1), col=(j+1))
            fig.update_layout(height=500)
            
    fig.for_each_annotation(lambda a: a.update(font_size=18))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Cluster_Assignment=", ""), y=1.05),
                            selector={'xanchor': 'center'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("type=", "")))

    # annotate any cluster that is significantly different from all other clusters
    # using fisher exact test to compare each cluster to all other clusters
    fisher_p_df = pd.DataFrame()
    for col1, col2 in zip(['non-coding', 'synonymous'], ['coding', 'non-synonymous']):
        for idx in mut_counts_df.index:
            odds_ratio, p_val = ss.fisher_exact([[mut_counts_df.loc[idx, col1], mut_counts_df[col1].drop(idx).sum()],
                                                 [mut_counts_df.loc[idx, col2], mut_counts_df[col2].drop(idx).sum()]])
            fisher_p_df.loc[idx, col2] = p_val

    cluster_sig = (fisher_p_df < 0.05).any(axis=1)
    for idx in cluster_sig[cluster_sig].index:
        fig.for_each_annotation(lambda a: a.update(text=f'<b>{idx}*<b>', font_color='red'),
                                selector={'text': str(idx)})
    # todo add indication of which category is significantly different

    return [fig]


def classify_mut(variant_class):
    if variant_class in ['lincRNA', 'RNA', 'IGR', "3'UTR", "5'UTR", 'Intron', "5'Flank", "3'Flank", 'intronic']:
        return 'non-coding'
    elif variant_class in ['Silent', 'syn']:
        return 'synonymous'
    else:
        return 'non-synonymous'
