"""CNVPlotComponent.py module

Interactive CNV Plot with mutation multiplicity scatterplot

Mutation scatter interactive with mutation table

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

from cnv_suite.visualize import plot_acr_subplots, update_cnv_color_absolute, update_cnv_scatter_sigma_toggle, plot_acr_interactive
from cnv_suite.utils import calc_cn_levels, apply_segment_data_to_df, get_segment_interval_trees, switch_contigs
from AnnoMate.DataTypes.PatientSampleData import PatientSampleData


def gen_cnv_plot_app_component():
    """Generate CNV Plot app component"""
    return AppComponent(
        'CNV Plot',
        layout=gen_cnv_plot_layout(),
        callback_input=[
            Input('sample-selection-checklist', 'value'),
            Input('sigma_checklist', 'value'),
            Input('cnv-color-radioitem', 'value'),
            Input('absolute-cnv-box', 'value'),  # todo implement as switch, not checkbox
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
            State('mutation-selected-ids', 'value'),  # selected rows regardless of filtering
            State('mutation-filtered-ids', 'value')  # all rows in table after filtering
        ],
        new_data_callback=gen_absolute_components,
        internal_callback=internal_gen_absolute_components
    )

def gen_cnv_plot_layout():
    """Generate CNV Plot Component Layout"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='cnv_plot',
                    figure=go.Figure()
                ),
            ],
            width=10),
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
                    options=['Blue/Red', 'Difference', 'Cluster', 'Clonal/Subclonal', 'Black'],
                    value='Difference',
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
    """Calculate error for mutation scatter error bars"""
    if alt == 0:
        return 0
    else:
        return (beta.ppf(percentile, alt, ref) - alt / (alt + ref)) / purity

def gen_mut_scatter(maf_df, mut_sigma, sample):
    """Generate mutation scatterplot trace.

    Parameters
    ----------
    maf_df
        DataFrame from maf_fn filtered by the mutation table filtering dropdowns
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
                     'Cluster: %{customdata[4]:d}',
        showlegend=False
    )

    return mut_scatter


def gen_preloaded_cnv_plot(participant_df, participant_id, samples_df, output_dir):
    """Generate a CNV Plot to be stored in a pickle file

    Parameters
    ----------
    participant_df: pd.DataFrame
        Participant level DataFrame
    participant_id: str
        name of the participant
    samples_df: pd.DataFrame
        Sample level DataFrame
    output_dir: str
        Directory in which to save pre-computed pickle files

    Returns
    -------
    """
    sample_id_list = []
    cnv_seg_list = []
    sample_cnv_series = pd.Series()
    for sample_id in samples_df[samples_df['participant_id'] == participant_id].index:
        cnv_seg_df = pd.read_csv(samples_df.loc[sample_id, 'cnv_seg_fn'], sep='\t')
        cnv_plot, cnv_seg_df_mod, start_trace, end_trace = plot_acr_interactive(cnv_seg_df, csize, purity=samples_df.loc[sample_id, 'wxs_purity'], ploidy=samples_df.loc[sample_id, 'wxs_ploidy'])

        output_fn = f'{output_dir}/cnv_figs/{sample_id}.cnv_fig.pkl'
        pickle.dump([cnv_plot, cnv_seg_df_mod, start_trace, end_trace], open(output_fn, 'wb'))
        sample_id_list.append(sample_id)
        cnv_seg_list.append(cnv_seg_df_mod)
        sample_cnv_series.loc[sample_id] = output_fn

    participant_maf_series = gen_preloaded_maf(participant_df, participant_id, samples_df, cnv_seg_list, sample_id_list,
                                               output_dir)

    return sample_cnv_series, participant_maf_series


def gen_preloaded_maf(participant_df, participant_id, sample_df, cnv_segs, sample_ids, output_dir):
    maf_df = pd.read_csv(participant_df.loc[participant_id, 'maf_fn'], sep='\t')
    start_pos = maf_df.columns[maf_df.columns.isin(['Start_position', 'Start_Position'])][0]
    alt = maf_df.columns[maf_df.columns.isin(['Tumor_Seq_Allele2', 'Tumor_Seq_Allele'])][0]
    sample_id_col = maf_df.columns[maf_df.columns.isin(['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id', 'Sample_id'])][0]
    maf_df['id'] = maf_df.apply(lambda x: get_unique_identifier(x, start_pos=start_pos, alt=alt), axis=1)

    maf_df['Sample_ID'] = maf_df[sample_id_col]
    # maf_df = switch_contigs(maf_df)
    # .replace giving: Series.replace cannot use dict-like to_replace and non-None value ?? todo
    maf_df['Chromosome'] = maf_df['Chromosome'].apply(lambda x: '23' if x == 'X' else '24' if x == 'Y' else x)

    for i, seg_df in enumerate(cnv_segs):
        seg_df['Sample_ID'] = sample_ids[i]

    seg_trees = get_segment_interval_trees(pd.concat(cnv_segs))
    maf_df = apply_segment_data_to_df(maf_df, seg_trees)
    maf_df.set_index('id', inplace=True, drop=False)

    purity = sample_df['wxs_purity'].astype(float).to_dict()
    ploidy = sample_df['wxs_ploidy'].astype(float).to_dict()
    c_values = [calc_cn_levels(pur, plo) for pur, plo in zip(purity.values(), ploidy.values())]
    c_0 = {sample: c[0] for sample, c in zip(purity.keys(), c_values)}
    c_delta = {sample: c[1] for sample, c in zip(purity.keys(), c_values)}

    maf_df['purity'] = maf_df['Sample_ID'].replace(purity)
    maf_df['ploidy'] = maf_df['Sample_ID'].replace(ploidy)
    maf_df['c_0'] = maf_df['Sample_ID'].replace(c_0)
    maf_df['c_delta'] = maf_df['Sample_ID'].replace(c_delta)

    maf_df['Chromosome'] = maf_df['Chromosome'].astype(int)
    if 'Cluster_Assignment' in maf_df:
        maf_df['cluster_color'] = maf_df['Cluster_Assignment'].astype(int).apply(lambda x: cluster_color(x))

    c_size_cumsum = np.cumsum([0] + list(csize.values()))
    maf_df['x_loc'] = maf_df.apply(lambda x: c_size_cumsum[x['Chromosome'] - 1] + x['Start_position'], axis=1)
    maf_df['VAF'] = maf_df['t_alt_count'] / (maf_df['t_alt_count'] + maf_df['t_ref_count'])

    maf_df['mu_major_adj'] = (maf_df['mu_major'] - maf_df['c_0']) / maf_df['c_delta']
    maf_df['mu_minor_adj'] = (maf_df['mu_minor'] - maf_df['c_0']) / maf_df['c_delta']
    maf_df['multiplicity_ccf'] = maf_df.apply(
        lambda x: x.VAF * (x['purity'] * (x.mu_major_adj + x.mu_minor_adj) + 2 * (1 - x['purity'])) / x['purity'], axis=1)
    # calculate error bars for mutations
    maf_df['error_top'] = maf_df.apply(
        lambda x: calculate_error(x.t_alt_count, x.t_ref_count, x['purity'], 0.975), axis=1)
    maf_df['error_bottom'] = maf_df.apply(
        lambda x: -1 * calculate_error(x.t_alt_count, x.t_ref_count, x['purity'], 0.025), axis=1)

    output_fn = f'{output_dir}/maf_df/{participant_id}.maf_df.pkl'
    pickle.dump(maf_df, open(output_fn, 'wb'))

    return pd.Series(output_fn, [participant_id])


def gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, preprocess_data_dir):
    """Generate CNV Plot with all customizations.

    Parameters
    ----------
    df
        participant level DataFrame
    idx
    sample_selection
        sample selection checkbox value
    sigmas
        sigma checkbox value
    color
        color checkbox value
    absolute
        absolute CN checkbox value
    selected_mutation_rows
        rows selected in the mutation table, None if none selected
    filtered_mutation_rows
        rows filtered in the mutations table, None if none selected
    samples_df
        sample level dataframe
    preprocess_data_dir
        directory path in which preprocessed CNV pickle files are stored

    Returns
    -------
    cnv_plot : make_subplots()
    sample_list : list of str
        sample checkbox options
    sample_selection_corrected
        first two selections in the sample selection checkbox

    Notes
    -----
    No mutation scatter plot if no purity and ploidy in data
    """
    sample_list = samples_df[samples_df['participant_id'] == idx].sort_values('collection_date_dfd').index.tolist()
    # start with only first sample selected
    sample_selection_corrected = [sample_list[0]] if sample_selection == [] else \
        [s for s in sample_list if s in sample_selection]  # get correct order

    sigmas_val = 'Show CNV Sigmas' in sigmas
    absolute_val = 'Display Absolute CN' in absolute

    # collect Figures in list
    fig_list = []
    for i, sample_id in enumerate(sample_selection_corrected):
        cnv_plot, this_seg_df, start_trace, end_trace = pickle.load(
            open(samples_df.loc[sample_id, 'cnv_fig_pickle'], "rb"))
        update_cnv_color_absolute(cnv_plot, this_seg_df, absolute_val, color, start_trace, end_trace)

        fig_list.append(cnv_plot)

    cnv_subplots_fig = plot_acr_subplots(fig_list, 'Copy Number Plots', sample_selection_corrected, csize)
    update_cnv_scatter_sigma_toggle(cnv_subplots_fig, sigmas_val)

    maf_df = pickle.load(open(df.loc[idx, 'maf_df_pickle'], "rb"))
    if selected_mutation_rows:
        maf_df = maf_df.loc[selected_mutation_rows]
    elif filtered_mutation_rows:
        maf_df = maf_df.loc[filtered_mutation_rows]
    # else (if all mutations in table are filtered out and none selected): use all mutations

    for i, sample_id in enumerate(sample_selection_corrected):
        sample_maf_df = maf_df[maf_df['Sample_ID'] == sample_id]
        cnv_subplots_fig.add_trace(gen_mut_scatter(sample_maf_df, sigmas_val, sample_id), row=i+1, col=1)

    return [
        cnv_subplots_fig,
        sample_list,
        sample_selection_corrected
    ]

def gen_absolute_components(
    data: PatientSampleData, idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, selected_mutation_rows, filtered_mutation_rows, preprocess_data_dir):
    """Absolute components callback function with parameters being the callback inputs/states and returns being callback outputs."""
    df = data.participant_df
    samples_df = data.sample_df

    # when changing participants, show all mutations at first
    # the filtered and selected mutations input to this function are from the old (previous participant's) MutationTable
    filtered_mutation_rows = None
    selected_mutation_rows = None

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
