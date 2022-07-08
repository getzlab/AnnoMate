import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import dash_cytoscape as cyto
import functools
import itertools
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
from scipy.stats import beta

import cnv_suite
from cnv_suite.visualize import plot_acr_interactive, update_cnv_scatter_cn, update_cnv_scatter_color, update_cnv_scatter_sigma_toggle
from cnv_suite import calc_avg_cn, calc_absolute_cn, calc_cn_levels, return_seg_data_at_loci, apply_segment_data_to_df, get_segment_interval_trees, switch_contigs

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
from JupyterReviewer.AppComponents.PatientReviewerLayout import PatientReviewerLayout
#from JupyterReviewer.lib.plot_cnp import plot_acr_interactive

def validate_purity(x):
    return (x >= 0) and (x <= 1)

def validate_ploidy(x):
    return x >= 0

def gen_clinical_data_table(df, idx, cols):
    r=df.loc[idx]
    return [dbc.Table.from_dataframe(r[cols].to_frame().reset_index())]

start_pos = 'Start_position' or 'Start_Position'
end_pos = 'End_position' or 'End_Position'
protein_change = 'Protein_change' or 'Protein_Change'
t_ref_count = 't_ref_count' or 't_ref_count_pre_forecall'
t_alt_count = 't_alt_count' or 't_alt_count_pre_forecall'

default_maf_cols = [
    'Hugo_Symbol',
    'Chromosome',
    start_pos,
    end_pos,
    protein_change,
    'Variant_Classification',
    t_ref_count,
    t_alt_count,
    'n_ref_count',
    'n_alt_count'
]

maf_cols_options = []
maf_cols_value = []
hugo_symbols = []
variant_classifications = []
cluster_assignments = []

treatment_category_colors = {
    'Chemotherapy': 'MidnightBlue',
    'Hormone/Endocrine therapy': 'MistyRose',
    'Precision/Targeted therapy': 'Plum'
}

def get_hex_string(c):
    return '#{:02X}{:02X}{:02X}'.format(*c)

def cluster_color(v):
    phylogic_color_list = [[166, 17, 129],
                           [39, 140, 24],
                           [103, 200, 243],
                           [248, 139, 16],
                           [16, 49, 41],
                           [93, 119, 254],
                           [152, 22, 26],
                           [104, 236, 172],
                           [249, 142, 135],
                           [55, 18, 48],
                           [83, 82, 22],
                           [247, 36, 36],
                           [0, 79, 114],
                           [243, 65, 132],
                           [60, 185, 179],
                           [185, 177, 243],
                           [139, 34, 67],
                           [178, 41, 186],
                           [58, 146, 231],
                           [130, 159, 21],
                           [161, 91, 243],
                           [131, 61, 17],
                           [248, 75, 81],
                           [32, 75, 32],
                           [45, 109, 116],
                           [255, 169, 199],
                           [55, 179, 113],
                           [34, 42, 3],
                           [56, 121, 166],
                           [172, 60, 15],
                           [115, 76, 204],
                           [21, 61, 73],
                           [67, 21, 74],  # Additional colors, uglier and bad
                           [123, 88, 112],
                           [87, 106, 46],
                           [37, 66, 58],
                           [132, 79, 62],
                           [71, 58, 32],
                           [59, 104, 114],
                           [46, 107, 90],
                           [84, 68, 73],
                           [90, 97, 124],
                           [121, 66, 76],
                           [104, 93, 48],
                           [49, 67, 82],
                           [71, 95, 65],
                           [127, 85, 44],  # even more additional colors, gray
                           [88, 79, 92],
                           [220, 212, 194],
                           [35, 34, 36],
                           [200, 220, 224],
                           [73, 81, 69],
                           [224, 199, 206],
                           [120, 127, 113],
                           [142, 148, 166],
                           [153, 167, 156],
                           [162, 139, 145],
                           [0, 0, 0]]  # black
    colors_dict = {str(i): get_hex_string(c) for i, c in enumerate(phylogic_color_list)}

    return colors_dict[str(v)]

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

def gen_style_data_conditional(df, custom_colors):
    style_data_conditional = []

    if 'Cluster_Assignment' in maf_cols_value:
        for n in df.Cluster_Assignment.unique():
            style_data_conditional.append(style_data_format('Cluster_Assignment', n, color=cluster_color(n)))

    if 'functional_effect' in maf_cols_value:
        style_data_conditional.extend([
            style_data_format('functional_effect', 'Likely Loss-of-function', backgroundColor='DarkOliveGreen'),
            style_data_format('functional_effect', 'Likely Gain-of-function', backgroundColor='DarkSeaGreen')
        ])

    if 'oncogenic' in maf_cols_value:
        style_data_conditional.append(style_data_format('oncogenic', 'Likely Oncogenic', backgroundColor='DarkOliveGreen'))

    if 'dbNSFP_Polyphen2_HDIV_ann' in maf_cols_value:
        style_data_conditional.append(style_data_format('dbNSFP_Polyphen2_HDIV_ann', 'D', backgroundColor='FireBrick'))

    if custom_colors != []:
        for list in custom_colors:
            style_data_conditional.append(style_data_format(list[0], list[1], list[2], list[3]))

    return style_data_conditional

def gen_maf_columns(df, idx, cols, hugo, variant, cluster):
    maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    #maf_df = pd.read_csv('~/Broad/JupyterReviewer/example_notebooks/example_data/all_mut_ccfs_maf_annotated_w_cnv_single_participant.txt', sep='\t')
    maf_cols_options = (list(maf_df))

    for col in default_maf_cols:
        if col in maf_cols_options and col not in maf_cols_value:
            maf_cols_value.append(col)

    for col in cols:
        if col in maf_cols_options and col not in maf_cols_value:
            maf_cols_value.append(col)

    for symbol in maf_df.Hugo_Symbol.unique():
        if symbol not in hugo_symbols:
            hugo_symbols.append(symbol)

    for classification in maf_df.Variant_Classification.unique():
        if classification not in variant_classifications:
            variant_classifications.append(classification)

    for n in maf_df.Cluster_Assignment.unique():
        if n not in cluster_assignments:
            cluster_assignments.append(n)

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
        sorted(cluster_assignments),
        filtered_maf_df
    ]

def gen_maf_table(df, idx, cols, hugo, table_size, variant, cluster, custom_colors):
    maf_df, maf_cols_options, maf_cols_value, hugo_symbols, variant_classifications, cluster_assignments, filtered_maf_df = gen_maf_columns(df, idx, cols, hugo, variant, cluster)

    return [
        maf_cols_options,
        maf_cols_value,
        dash_table.DataTable(
            data=filtered_maf_df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'selectable': True} for i in maf_cols_value],
            filter_action='native',
            sort_action='native',
            row_selectable='single',
            column_selectable='multi',
            page_action='native',
            page_current=0,
            page_size=table_size,
            style_data_conditional=gen_style_data_conditional(filtered_maf_df, custom_colors)
        ),
        hugo_symbols,
        variant_classifications,
        cluster_assignments
    ]

def internal_gen_maf_table(df, idx, cols, hugo, table_size, variant, cluster, custom_colors):
    maf_df, maf_cols_options, maf_cols_value, hugo_symbols, variant_classifications, cluster_assignments, filtered_maf_df = gen_maf_columns(df, idx, cols, hugo, variant, cluster)

    return [
        maf_cols_options,
        cols,
        dash_table.DataTable(
                data=filtered_maf_df.to_dict('records'),
                columns=[{'name': i, 'id': i, 'selectable': True} for i in cols],
                filter_action='native',
                sort_action='native',
                row_selectable='single',
                column_selectable='multi',
                page_action='native',
                page_current=0,
                page_size=table_size,
                style_data_conditional=gen_style_data_conditional(filtered_maf_df, custom_colors)
        ),
        hugo_symbols,
        variant_classifications,
        cluster_assignments
    ]

def gen_ccf_plot(df, idx, time_scaled, biospecimens_fn):
    maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    samples = maf_df.drop_duplicates('Sample_ID').Sample_ID.tolist()

    biospecimens_df = pd.read_csv(biospecimens_fn, sep='\t').set_index('participant_id').loc[idx]
    biospecimens_df = biospecimens_df.set_index('collaborator_sample_id')
    biospecimens_df = biospecimens_df.loc[[sample for sample in samples if sample in biospecimens_df.index]]

    timing_data = {}
    for sample in samples:
        if sample in biospecimens_df.index:
            timing_data[sample] = biospecimens_df.loc[sample,'collection_date_dfd']
        else:
            timing_data[sample] = 0

    if 'Time Scaled' in time_scaled:
        scatter_x = 'dfd'
        rect_x = 5
    else:
        scatter_x = 'order'
        rect_x = 6

    cluster_ccfs = pd.read_csv(df.loc[idx, 'cluster_ccfs_fn'], sep='\t')
    mut_ccfs = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    cluster_df = cluster_ccfs[['Cluster_ID', 'Sample_ID', 'postDP_ccf_mean', 'postDP_ccf_CI_low', 'postDP_ccf_CI_high']].copy()

    cluster_df.loc[:, 'dfd'] = [int(timing_data[sample]) for sample in cluster_ccfs['Sample_ID']]
    samples_in_order = sorted(timing_data.keys(), key=lambda k: int(timing_data[k]))
    ordered_samples_dict = {s: o for s, o in zip(samples_in_order, np.arange(len(samples_in_order)))}
    cluster_df.loc[:, 'order'] = [ordered_samples_dict[s] for s in cluster_ccfs['Sample_ID']]

    treatments_df = pd.read_csv('~/Broad/JupyterReviewer/example_notebooks/example_data/treatments.tsv', sep='\t').set_index('participant_id').loc[idx]
    #treatments_df = treatments_df[treatment['stop_date_dfd'] >= timing_data[samples_in_order[0]] or treatment['start_date_dfd'] <= timing_data[samples_in_order[-1]] for treatment in treatments_df]
    treatments_in_frame_df = pd.DataFrame()
    for start, stop in zip(treatments_df.start_date_dfd, treatments_df.stop_date_dfd):
        if stop >= timing_data[samples_in_order[0]] and start <= timing_data[samples_in_order[-1]]:
            treatments_in_frame_df = pd.concat([treatments_df[treatments_df.start_date_dfd == start], treatments_in_frame_df])

    mut_count_dict = mut_ccfs.drop_duplicates([
        'Patient_ID',
        'Hugo_Symbol',
        'Chromosome',
        'Start_position',
        'Cluster_Assignment'
    ]).groupby('Cluster_Assignment').count()['Patient_ID'].to_dict()

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
    #ccf_plot.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400, width=600)
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
            #hovertemplate = 'treatments',
            showlegend=False
        ),
        row=2, col=1
    )

    for start, stop, drug, drug_combo, category, stop_reason, post_status in zip(treatments_in_frame_df.start_date_dfd, treatments_in_frame_df.stop_date_dfd, treatments_in_frame_df.drugs, treatments_in_frame_df.drug_combination, treatments_df.categories, treatments_in_frame_df.stop_reason, treatments_in_frame_df.post_status):
        drugs=drug
        if pd.isna(drug):
            drugs=drug_combo

        ccf_plot.add_trace(
            go.Scatter(
                x=[max(start, timing_data[samples_in_order[0]]), min(stop, timing_data[samples_in_order[-1]])],
                y=[0,0],
                line_width=20,
                line_color=treatment_category_colors[category],
                fill='toself',
                hovertemplate = f'Treatment Regimen: {drugs}; '
                    f'Stop Reason: {stop_reason}; '
                    f'Post Status: {post_status}',
                showlegend=False
            ),
            row=2, col=1
        )
        ccf_plot.add_vline(
            x=max(start, timing_data[samples_in_order[0]]),
            line_width=2,
            line_color='black',
            row=2, col=1
        )
        ccf_plot.add_vline(
            x=ccf_plot.add_vline(
                x=min(stop, timing_data[samples_in_order[-1]]),
                line_width=2,
                line_color='black',
                row=2, col=1
            ),
            line_width=2,
            line_color='black',
            row=2, col=1
        )

    ccf_plot.update_yaxes(row=2, visible=False)
    ccf_plot.update_xaxes(row=1, visible=False, showticklabels=False)

    return ccf_plot

def gen_stylesheet(cluster_list, color_list):
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
                'background-color': color_list[int(node) - 1]
            }
        })
        stylesheet.append({
            'selector': ('edge[target = "%s"]' % f'cluster_{node}'),
            'style': {
                'line-color': color_list[int(node) - 1]
            }
        })

    return stylesheet

possible_trees = []
all_trees = []
clusters = {}
cluster_count = {}

def gen_driver_edge_labels(drivers, clusters, cluster):
    label = ''
    for driver in drivers.drivers:
        if driver in clusters[cluster]:
            label += ('%s \n' % driver)

    return label

def gen_phylogic_tree(df, idx, tree_num, drivers_fn):
    tree_df = pd.read_csv(df.loc[idx, 'build_tree_posterior_fn'], sep='\t')
    maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    maf_df.drop_duplicates(subset='Start_position', inplace=True)
    if drivers_fn:
        drivers = pd.read_csv(f'~/Broad/JupyterReviewer/{drivers_fn}')

    possible_trees = []
    all_trees = []
    clusters = {}
    cluster_count = {}

    trees = tree_df.loc[:, 'edges']
    for i, tree in enumerate(trees):
        all_trees.append(tree.split(','))
        possible_trees.append(f'Tree {i+1} ({tree_df.n_iter[i]})')

    for i in range(len(cluster_assignments)):
        clusters[cluster_assignments[i]] = [hugo for clust, hugo in zip(maf_df.Cluster_Assignment, maf_df.Hugo_Symbol) if clust == cluster_assignments[i]]

    for clust in clusters:
        cluster_count[clust] = len(clusters[clust])

    edges = all_trees[tree_num]

    cluster_list = []
    for i in edges:
        new_list = i.split('-')
        for j in new_list:
            if (j !='None') & (j not in cluster_list):
                cluster_list.append(j)

    color_list=[]
    for node in cluster_list:
        color_list.append(cluster_color(node))

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
        edges = [{'data': {'source': 'normal', 'target': 'cluster_1', 'label': f'{cluster_count[1]}\n{gen_driver_edge_labels(drivers, clusters, 1)}'}}]
        edges.extend([
            {'data': {'source': f'cluster_{edge[0]}', 'target': f'cluster_{edge[1]}', 'label': f'{cluster_count[edge[1]]}\n{gen_driver_edge_labels(drivers, clusters, edge[1])}'}}
            for edge in edges_list
        ])
    else:
        edges = [{'data': {'source': 'normal', 'target': 'cluster_1', 'label': str(cluster_count[1])}}]
        edges.extend([
            {'data': {'source': f'cluster_{edge[0]}', 'target': f'cluster_{edge[1]}', 'label': str(cluster_count[edge[1]])}}
            for edge in edges_list
        ])

    elements = nodes + edges

    stylesheet = gen_stylesheet(cluster_list, color_list)

    return [
        cyto.Cytoscape(
            id='phylogic-tree',
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

def gen_phylogic_graphics(df, idx, time_scaled, chosen_tree, mutation, drivers_fn, biospecimens_fn):
    ccf_plot = gen_ccf_plot(df, idx, time_scaled, biospecimens_fn)
    tree, possible_trees = gen_phylogic_tree(df, idx, 0, drivers_fn)

    return [ccf_plot, possible_trees, possible_trees[0], tree]

def internal_gen_phylogic_graphics(df, idx, time_scaled, chosen_tree, mutation, drivers_fn, biospecimens_fn):
    tree_num = 0
    for n in chosen_tree.split():
        if n.isdigit():
            tree_num = int(n)

    ccf_plot = gen_ccf_plot(df, idx, time_scaled, biospecimens_fn)
    tree, possible_trees = gen_phylogic_tree(df, idx, tree_num-1, drivers_fn)

    return [ccf_plot, possible_trees, chosen_tree, tree]

def calculate_error(alt, ref, purity, percentile):
    if alt == 0:
        return 0
    else:
        return (beta.ppf(percentile, alt, ref) - alt / (alt + ref)) / purity

def gen_mut_scatter(maf_df, mut_sigma, sample):
    mut_scatter = go.Scatter(
        x=maf_df['x_loc'],
        y=maf_df['multiplicity_ccf'],
        mode='markers',
        marker_size=5,
        marker_color=maf_df['cluster_color'],
        name=f'Mutations ({sample})',
        error_y=dict(
            type='data',
            array=maf_df['error_top'],
            arrayminus=maf_df['error_bottom'],
            color='gray',
            visible=mut_sigma,
            width=0
        ),
        customdata=np.stack((
            maf_df['Hugo_Symbol'].tolist(),
            maf_df['Chromosome'].tolist(),
            maf_df['Start_position'].tolist(),
            maf_df['VAF'].tolist(),
            maf_df['Cluster_Assignment'].tolist(),
            maf_df['Variant_Type'].tolist(),
            maf_df['Variant_Classification'].tolist(),
            maf_df['Protein_change']),
            axis=-1
        ),
        hovertemplate='<extra></extra>' +
                     'Gene: %{customdata[0]} %{customdata[1]}:%{customdata[2]} <br>' +
                     'Variant: %{customdata[5]}, %{customdata[6]} <br>' +
                     'Protein Change: %{customdata[7]} <br>' +
                     'Multiplicity: %{y:.3f} <br>' +
                     'VAF: %{customdata[3]:.3f} <br>' +
                     'Cluster: %{customdata[4]:d}')

    return mut_scatter

def gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, clusters, hugo, variant_classification, samples_fn):
    csize = {'1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276, '5': 180915260,
            '6': 171115067, '7': 159138663, '8': 146364022, '9': 141213431, '10': 135534747,
            '11': 135006516, '12': 133851895, '13': 115169878, '14': 107349540, '15': 102531392,
            '16': 90354753, '17': 81195210, '18': 78077248, '19': 59128983, '20': 63025520,
            '21': 48129895, '22': 51304566, '23': 156040895, '24': 57227415}

    all_samples_df = pd.read_csv(samples_fn)
    all_samples_df.set_index('Sample_ID', inplace=True)

    original_maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    original_maf_df['Chromosome'] = original_maf_df['Chromosome'].astype(int)

    maf_df = original_maf_df.copy()
    if hugo:
        maf_df = maf_df[maf_df.Hugo_Symbol.isin(hugo)]
    if variant_classification:
        maf_df = maf_df[maf_df.Variant_Classification.isin(variant_classification)]
    if clusters:
        maf_df = maf_df[maf_df.Cluster_Assignment.isin(clusters)]

    sample_list = all_samples_df[all_samples_df['participant_id'] == idx].index.tolist()
    # restrict sample selection to only two samples at a time
    sample_selection_corrected = [sample_list[0]] if sample_selection == [] else sample_selection[:2]

    sigmas_val = False
    if 'Show CNV Sigmas' in sigmas:
        sigmas_val = True

    if color == 'Differential':
        segment_colors = 'difference'
    elif color == 'Cluster':
        segment_colors = 'cluster'
    # unsure about clonal/subclonal
    else:
        segment_colors = color

    seg_df = []
    for sample_id in sample_selection_corrected:
        this_seg_df = pd.read_csv(all_samples_df.loc[sample_id, 'absolute_fn'], sep='\t')
        this_seg_df['Sample_ID'] = sample_id
        seg_df.append(this_seg_df)

    seg_trees = get_segment_interval_trees(pd.concat(seg_df))
    #maf_df = apply_segment_data_to_df(maf_df, seg_trees)

    c_size_cumsum = np.cumsum([0] + list(csize.values()))
    maf_df['x_loc'] = maf_df.apply(lambda x: c_size_cumsum[x['Chromosome'] - 1] + x['Start_position'], axis=1)
    maf_df['cluster_color'] = maf_df['Cluster_Assignment'].apply(lambda x: cluster_color(x))
    maf_df['VAF'] = maf_df['t_alt_count'] / (maf_df['t_alt_count'] + maf_df['t_ref_count'])

    cnv_plot = make_subplots(len(sample_selection_corrected), 1)
    for i, sample_id in enumerate(sample_selection_corrected):
        plot_acr_interactive(seg_df[i], cnv_plot, csize, segment_colors=segment_colors, sigmas=sigmas_val, row=i)

        purity = all_samples_df.loc[sample_id, 'wxs_purity']
        ploidy = all_samples_df.loc[sample_id, 'wxs_ploidy']
        c_0, c_delta = calc_cn_levels(purity, ploidy)

        this_maf_df = maf_df[maf_df['Sample_ID'] == sample_id]
        this_seg_df = seg_df[i]
        if 'Display Absolute CN' in absolute:
            this_maf_df['mu_major_adj'] = (this_seg_df['mu.major'] - c_0) / c_delta
            this_maf_df['mu_minor_adj'] = (this_seg_df['mu.minor'] - c_0) / c_delta
        else:
            this_maf_df['mu_major_adj'] = this_seg_df['mu.major']
            this_maf_df['mu_minor_adj'] = this_seg_df['mu.minor']

        this_maf_df['multiplicity_ccf'] = this_maf_df.apply(
            lambda x: x.VAF * (purity * (x.mu_major_adj + x.mu_minor_adj) + 2 * (1 - purity)) / purity, axis=1
        )

         # calculate error bars for mutations
        this_maf_df['error_top'] = this_maf_df.apply(
            lambda x: calculate_error(x.t_alt_count, x.t_ref_count, purity, 0.975), axis=1)
        this_maf_df['error_bottom'] = this_maf_df.apply(
            lambda x: -1 * calculate_error(x.t_alt_count, x.t_ref_count, purity, 0.025), axis=1)

        cnv_plot.add_trace(gen_mut_scatter(this_maf_df, sigmas_val, sample_id), row=i+1, col=1)

    return [
        cnv_plot,
        sample_list,
        sample_selection_corrected
    ]

def gen_absolute_components(df, idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, clusters, hugo, variant_classification, samples_fn):
    cnv_plot, sample_list, sample_selection = gen_cnv_plot(df, idx, [], sigmas, color, absolute, clusters, hugo, variant_classification, samples_fn)
    button_clicks = None

    return [
        cnv_plot,
        sample_list,
        sample_selection,
        button_clicks
    ]

def internal_gen_absolute_components(df, idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, clusters, hugo, variant_classification, samples_fn):
    if button_clicks != None:
        cnv_plot, sample_list, sample_selection = gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, clusters, hugo, variant_classification, samples_fn)
        button_clicks = None

    return [
        cnv_plot,
        sample_list,
        sample_selection,
        button_clicks
    ]

class PatientReviewer(ReviewerTemplate, PatientReviewerLayout):

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
            'tree': ReviewDataAnnotation('text'),
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
    def gen_review_app(self, biospecimens_fn, custom_colors=[], drivers_fn=None, samples_fn=None) -> ReviewDataApp:
        app = ReviewDataApp()

        app.add_component(AppComponent(
            'Clinical Data',
            html.Div(
                id='clinical-data-component',
                children=dbc.Table.from_dataframe(df=pd.DataFrame())
            ),
            callback_output=[Output('clinical-data-component', 'children')],
            new_data_callback=gen_clinical_data_table
        ), cols=['gender', 'age_at_diagnosis', 'vital_status', 'death_date_dfd'])

        app.add_component(AppComponent(
            'Mutations',
            layout=PatientReviewerLayout.gen_mutation_table_layout(),

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
                Output('variant-classification-dropdown', 'options'),
                Output('cluster-assignment-dropdown', 'options')
            ],
            new_data_callback=gen_maf_table,
            internal_callback=internal_gen_maf_table
        ), custom_colors=custom_colors)

        app.add_component(AppComponent(
            'Phylogic Graphics',
            layout=PatientReviewerLayout.gen_phylogic_components_layout(),

            callback_input=[
                Input('time-scale-checklist', 'value'),
                Input('tree-dropdown', 'value')
            ],
            callback_state_external=[
                State('mutation-table-component', 'children')
            ],
            callback_output=[
                Output('ccf-plot', 'figure'),
                Output('tree-dropdown', 'options'),
                Output('tree-dropdown', 'value'),
                Output('phylogic-tree-component', 'children')
            ],
            callback_states_for_autofill=[
                State('tree-dropdown', 'value')
            ],
            new_data_callback=gen_phylogic_graphics,
            internal_callback=internal_gen_phylogic_graphics
        ), drivers_fn=drivers_fn, biospecimens_fn=biospecimens_fn)

        app.add_component(AppComponent(
            'CNV Plot',
            layout=PatientReviewerLayout.gen_cnv_plot_layout(),
            callback_input=[
                Input('sample-selection-checklist', 'value'),
                Input('sigma_checklist', 'value'),
                Input('cnv-color-radioitem', 'value'),
                Input('absolute-cnv-box', 'value'),
                Input('cnv-button', 'n_clicks'),
                Input('cnv_plot', 'figure'),
                Input('sample-selection-checklist', 'options')
            ],
            callback_output=[
                Output('cnv_plot', 'figure'),
                Output('sample-selection-checklist', 'options'),
                Output('sample-selection-checklist', 'value'),
                Output('cnv-button', 'n_clicks')
            ],
            callback_state_external=[
                State('cluster-assignment-dropdown', 'value'),
                State('hugo-dropdown', 'value'),
                State('variant-classification-dropdown', 'value')
            ],
            new_data_callback=gen_absolute_components,
            internal_callback=internal_gen_absolute_components
        ), samples_fn=samples_fn)

        # app.add_component(AppComponent(
        #     'Purity Slider',
        #     html.Div(dcc.Slider(0, 1, 0.1, value=0.5, id='a-slider')),
        #     callback_output=[Output('a-slider', 'value')],
        #     callback_states_for_autofill=[State('a-slider', 'value')]
        # ))

        return app

    def gen_autofill(self):
        #self.add_autofill('Purity Slider', {'purity': State('a-slider', 'value')})
        self.add_autofill('Phylogic Graphics', {'tree': State('tree-dropdown', 'value')})
