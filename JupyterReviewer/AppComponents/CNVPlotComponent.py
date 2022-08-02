import pandas as pd
import numpy as np
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta
import pickle

from JupyterReviewer.ReviewDataApp import AppComponent
from JupyterReviewer.AppComponents.utils import cluster_color, get_unique_identifier

import cnv_suite
from cnv_suite.visualize import plot_acr_interactive, update_cnv_scatter_cn, update_cnv_scatter_color, update_cnv_scatter_sigma_toggle, calc_color, get_phylogic_color_scale
from cnv_suite.utils import calc_avg_cn, calc_absolute_cn, calc_cn_levels, return_seg_data_at_loci, apply_segment_data_to_df, get_segment_interval_trees, switch_contigs
from JupyterReviewer.DataTypes.PatientSampleData import PatientSampleData


def gen_cnv_plot_app_component():
    return AppComponent(
        'CNV Plot',
        layout=gen_cnv_plot_layout(),
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
            State('mutation-table', 'selected_row_ids'),  # selected rows regardless of filtering
            State('mutation-table', 'derived_virtual_row_ids')  # all rows in table after filtering (and sorting)
        ],
        new_data_callback=gen_absolute_components,
        internal_callback=internal_gen_absolute_components
    )

def gen_cnv_plot_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='cnv_plot',
                    figure=go.Figure()
                ),
            ], width=10),
            dbc.Col([
                html.H3('Customize Plot'),
                html.H5('Samples:'),
                dcc.Checklist(
                    id='sample-selection-checklist',
                    options=[],
                    value=[],
                    labelStyle={'display': 'block'}
                ),
                html.P(''),
                html.H5('Sigmas:'),
                dcc.Checklist(
                    id='sigma_checklist',
                    options=['Show CNV Sigmas'],
                    value=['Show CNV Sigmas']
                ),
                html.P(''),
                html.H5('Colors:'),
                dcc.RadioItems(
                    id='cnv-color-radioitem',
                    options=['Differential', 'Cluster', 'Red/Blue', 'Clonal/Subclonal'],
                    value='Differential',
                    labelStyle={'display': 'block'}
                ),
                html.P(''),
                html.H5('Scale:'),
                dcc.Checklist(
                    options=['Display Absolute CN'],
                    value=['Display Absolute CN'],
                    id='absolute-cnv-box'
                ),
                html.P(''),
                html.Button('Submit', id='cnv-button')
            ], width=2)
        ]),
    ])

csize = {'1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276, '5': 180915260,
        '6': 171115067, '7': 159138663, '8': 146364022, '9': 141213431, '10': 135534747,
        '11': 135006516, '12': 133851895, '13': 115169878, '14': 107349540, '15': 102531392,
        '16': 90354753, '17': 81195210, '18': 78077248, '19': 59128983, '20': 63025520,
        '21': 48129895, '22': 51304566, '23': 156040895, '24': 57227415}

def calculate_error(alt, ref, purity, percentile):
    if alt == 0:
        return 0
    else:
        return (beta.ppf(percentile, alt, ref) - alt / (alt + ref)) / purity

def gen_mut_scatter(maf_df, mut_sigma, sample):
    """Generate mutation scatterplot trace.

    Parameters
    ----------
    maf_df
        df from maf_fn filtered by the mutation table filtering dropdowns
    mut_sigma : bool
        sigmas value based on the sigma checkbox
    sample
        sample selected in the sample selection checkbox

    Returns
    -------
    mut_scatter : go.Scatter

    """
    mut_scatter = go.Scatter(
        x=maf_df['x_loc'],
        y=maf_df['multiplicity_ccf'],
        mode='markers',
        marker_size=5,
        marker_color=maf_df['cluster_color'] if 'Cluster_Assignment' in list(maf_df) else 'Black',
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
            maf_df['Cluster_Assignment'].tolist() if 'Cluster_Assignment' in list(maf_df) else [None]*len(maf_df),
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
                     'Cluster: %{customdata[4]:d}'
    )

    return mut_scatter

def gen_preloaded_cnv_plot(samples_df, sample):
    cnv_seg_df = pd.read_csv(samples_df.loc[sample, 'cnv_seg_fn'], sep='\t')

    cnv_plot = make_subplots()
    start_trace, end_trace = plot_acr_interactive(cnv_seg_df, cnv_plot, csize, segment_colors='difference', sigmas=True)

    return [cnv_plot, start_trace, end_trace]


def gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, preprocess_data_dir):
    """Generate CNV Plot with all customizations.

    Parameters
    ----------
    df
    idx
    sample_selection
        sample selection checkbox value
    sigmas
        sigma checkbox value
    color
        color checkbox value
    absolute
        absolute CN checkbox value
    samples_df
        sample dataframe passed into review_data_app as kwarg
    selected_mutation_rows

    filtered_mutation_rows
    preprocess_data_dir
        directory path in which preprocessed CNV pickle files are stored

    Returns
    -------
    cnv_plot : go.Figure
    sample_list : list of str
        sample checkbox options
    sample_selection_corrected
        sample checkbox value
    """

    maf_df = pd.read_csv(df.loc[idx, 'maf_fn'], sep='\t')
    start_pos = maf_df.columns[maf_df.columns.isin(['Start_position', 'Start_Position'])][0]
    alt = maf_df.columns[maf_df.columns.isin(['Tumor_Seq_Allele2', 'Tumor_Seq_Allele'])][0]
    sample_id_col = maf_df.columns[maf_df.columns.isin(['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id', 'Sample_id'])][0]
    maf_df['id'] = maf_df.apply(lambda x: get_unique_identifier(x, start_pos=start_pos, alt=alt), axis=1)
    maf_df.set_index('id', inplace=True, drop=False)

    if selected_mutation_rows:
        maf_df = maf_df.loc[selected_mutation_rows]
    elif filtered_mutation_rows:
        maf_df = maf_df.loc[filtered_mutation_rows]
    # else (if all mutations in table are filtered out and none selected): use all mutations

    sample_list = samples_df[samples_df['participant_id'] == idx].index.tolist()
    # restrict sample selection to only two samples at a time
    sample_selection_corrected = [sample_list[0]] if sample_selection == [] else sample_selection[:2]

    sigmas_val = 'Show CNV Sigmas' in sigmas

    seg_df = []
    for sample_id in sample_list:
        this_seg_df = pd.read_csv(samples_df.loc[sample_id, 'cnv_seg_fn'], sep='\t')
        if color == 'Differential':
            this_seg_df['color_bottom'], this_seg_df['color_top'] = calc_color(this_seg_df, 'mu.major', 'mu.minor')
        elif color == 'Cluster' and 'cluster_assignment' in this_seg_df.columns:
            phylogic_color_dict = get_phylogic_color_scale()
            this_seg_df['color_bottom'] = this_seg_df['cluster_assignment'].map(phylogic_color_dict)
            this_seg_df['color_top'] = this_seg_df['color_bottom']
        # unsure about clonal/subclonal
        else:
            this_seg_df['color_bottom'] = '#2C38A8'  # blue
            this_seg_df['color_top'] = '#E6393F'  # red
        this_seg_df['Sample_ID'] = sample_id
        seg_df.append(this_seg_df)

    maf_df['Sample_ID'] = maf_df[sample_id_col]
    maf_df = maf_df[maf_df.Sample_ID.isin(sample_list)]
    # maf_df = switch_contigs(maf_df)
    # .replace giving: Series.replace cannot use dict-like to_replace and non-None value ??
    maf_df['Chromosome'] = maf_df['Chromosome'].apply(lambda x: '23' if x == 'X' else '24' if x == 'Y' else x)

    seg_trees = get_segment_interval_trees(pd.concat(seg_df))
    maf_df = apply_segment_data_to_df(maf_df, seg_trees)

    maf_df['Chromosome'] = maf_df['Chromosome'].astype(int)
    if 'Cluster_Assignment' in list(maf_df):
        maf_df['Cluster_Assignment'] = maf_df['Cluster_Assignment'].astype(int)
        maf_df['cluster_color'] = maf_df['Cluster_Assignment'].apply(lambda x: cluster_color(x))

    c_size_cumsum = np.cumsum([0] + list(csize.values()))
    maf_df['x_loc'] = maf_df.apply(lambda x: c_size_cumsum[x['Chromosome'] - 1] + x['Start_position'], axis=1)
    maf_df['VAF'] = maf_df['t_alt_count'] / (maf_df['t_alt_count'] + maf_df['t_ref_count'])

    #cnv_plot = make_subplots(len(sample_selection_corrected), 1)
    for i, sample_id in enumerate(sample_selection_corrected):
        this_maf_df = maf_df[maf_df[sample_id_col] == sample_id]
        this_seg_df = seg_df[sample_list.index(sample_id)]

        cnv_plot, start_trace, end_trace = pickle.load(open(f'{preprocess_data_dir}/cnv_figs/{sample_id}.cnv_fig.pkl', "rb"))
        update_cnv_scatter_sigma_toggle(cnv_plot, sigmas_val)
        update_cnv_scatter_color(cnv_plot, this_seg_df['color_bottom'], this_seg_df['color_top'], start_trace, end_trace)
        #cnv_plot.add_trace(current_cnv_plot, row=i, col=1)

        if 'wxs_purity' in list(samples_df):
            purity = samples_df.loc[sample_id, 'wxs_purity']
            ploidy = samples_df.loc[sample_id, 'wxs_ploidy']
            c_0, c_delta = calc_cn_levels(purity, ploidy)

            if 'Display Absolute CN' in absolute:
                this_maf_df['mu_major_adj'] = (this_maf_df['mu_major'] - c_0) / c_delta
                this_maf_df['mu_minor_adj'] = (this_maf_df['mu_minor'] - c_0) / c_delta

                mu_major_adj, mu_minor_adj, sigma_major_adj, sigma_minor_adj = calc_absolute_cn(
                    this_seg_df['mu.major'], this_seg_df['mu.minor'], this_seg_df['sigma.major'], this_seg_df['sigma.minor'],
                    c_0, c_delta
                )

                update_cnv_scatter_cn(
                    cnv_plot,
                    mu_major_adj,
                    mu_minor_adj,
                    sigma_major_adj,
                    start_trace,
                    end_trace
                )
            else:
                this_maf_df['mu_major_adj'] = this_maf_df['mu_major']
                this_maf_df['mu_minor_adj'] = this_maf_df['mu_minor']

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

def gen_absolute_components(data: PatientSampleData, idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, selected_mutation_rows, filtered_mutation_rows, preprocess_data_dir):
    """Absolute components callback function with parameters being the callback inputs/states and returns being callback outputs."""
    df = data.participant_df
    samples_df = data.sample_df

    cnv_plot, sample_list, sample_selection = gen_cnv_plot(df, idx, [], sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, preprocess_data_dir)
    button_clicks = None

    return [
        cnv_plot,
        sample_list,
        sample_selection,
        button_clicks
    ]

def internal_gen_absolute_components(data: PatientSampleData, idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, selected_mutation_rows, filtered_mutation_rows, preprocess_data_dir):
    """Absolute components internal callback function with parameters being the callback inputs/states and returns being callback outputs."""
    df = data.participant_df
    samples_df = data.sample_df

    if button_clicks != None:
        cnv_plot, sample_list, sample_selection = gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, preprocess_data_dir)
        button_clicks = None

    return [
        cnv_plot,
        sample_list,
        sample_selection,
        button_clicks
    ]
