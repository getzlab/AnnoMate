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
import plotly.subplots
from matplotlib import patches
from scipy.stats import beta
import pickle
import functools

from AnnoMate.ReviewDataApp import AppComponent
from AnnoMate.AppComponents.utils import cluster_color, get_unique_identifier, freezeargs, cached_read_csv
from AnnoMate.DataTypes.PatientSampleData import PatientSampleData

from cnv_suite.visualize import plot_acr_subplots, update_cnv_color_absolute, \
    update_cnv_scatter_sigma_toggle, plot_acr_interactive, add_background
from cnv_suite.utils import calc_cn_levels, apply_segment_data_to_df, get_segment_interval_trees, switch_contigs



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
                html.Button('Reload Copy Number Plots', id='cnv-button')
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

def gen_mut_scatter(maf_df, mut_sigma, sample, maf_cluster_col, maf_hugo_col, maf_chromosome_col, maf_start_pos_col, maf_variant_class_col, maf_protein_change_col):
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
    # cluster_assignment = maf_df.columns[maf_df.columns.isin(['Cluster_Assignment', 'cluster'])][0]

    mut_scatter = go.Scatter(
        x=maf_df['x_loc'],
        y=maf_df['multiplicity_ccf'],
        mode='markers',
        marker_size=5,
        marker_color=maf_df['cluster_color'] if maf_cluster_col in list(maf_df) else 'Black',
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
            maf_df[maf_hugo_col].tolist(),
            maf_df[maf_chromosome_col].tolist(),
            maf_df[maf_start_pos_col].tolist(),
            maf_df['VAF'].tolist() if 'VAF' in maf_df else ['N/A']*len(maf_df),
            maf_df[maf_cluster_col].tolist() if maf_cluster_col in maf_df else ['N/A']*len(maf_df),
            maf_df['Variant_Type'].tolist() if 'Variant_Type' in maf_df else ['N/A']*len(maf_df),
            maf_df[maf_variant_class_col].tolist() if 'Variant_Classification' in maf_df else ['N/A']*len(maf_df),
            maf_df[maf_protein_change_col] if 'Protein_change' in maf_df else ['N/A']*len(maf_df)),
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

@freezeargs
@functools.lru_cache(maxsize=32)
def gen_seg_figure(cnv_seg_fn, csize, purity=None, ploidy=None):
    """Generate a CNV Plot from given seg file, purity, and ploidy

    Parameters
    ----------
    cnv_seg_fn: str
        Filename for sample seg file
    csize: dict
        Dictionary with contig sizes
    purity: float
        Tumor purity for this sample (optional)
    ploidy: float
        Tumor ploidy for this sample (optional)

    Returns
    -------
    (cnv_plot, cnv_seg_df_mod, start_trace, end_trace): (plotly.Figure, pd.DataFrame, int, int)
    """
    cnv_seg_df = cached_read_csv(cnv_seg_fn, sep='\t') 
    cnv_plot, cnv_seg_df_mod, start_trace, end_trace = plot_acr_interactive(cnv_seg_df, csize,
                                                                            purity=purity, ploidy=ploidy)

    return cnv_plot, cnv_seg_df_mod, start_trace, end_trace


@freezeargs
@functools.lru_cache(maxsize=16)
def gen_participant_cnv_and_maf(
    cnv_seg_filenames, 
    maf_fn, 
    sample_names, 
    csize, 
    purity_dict, 
    ploidy_dict, 
    maf_start_pos_col, 
    maf_sample_id_col, 
    maf_chromosome_col, 
    maf_cluster_col
):
    """Generate a CNV Plot to be stored in a pickle file

    Parameters
    ----------
    cnv_seg_filenames: list
        List of filenames for the sample cnv seg files
    maf_fn: str
        Filename for the participant maf file
    sample_names: list
        List of sample names for this participant
    csize: dict
        Dictionary with contig sizes
    purity_dict: dict
        Dictionary with purity values for this participant, keys given by sample_id
    ploidy_dict: dict
        Dictionary with ploidy values for this participant, keys given by sample_id

    Returns
    -------
    (participant_maf_df, cnv_plot_dict, cnv_seg_dict, trace_dict): (pd.DataFrame, dict, dict, dict)
    """
    cnv_seg_dict = {}
    cnv_plot_dict = {}
    trace_dict = {}
    for cnv_seg_fn, sample in zip(cnv_seg_filenames, sample_names):
        purity = purity_dict[sample]
        ploidy = ploidy_dict[sample]
        cnv_plot, cnv_seg_df_mod, start_trace, end_trace = gen_seg_figure(cnv_seg_fn, csize, purity=purity, ploidy=ploidy)
        cnv_seg_df_mod['Sample_ID'] = sample
        cnv_seg_dict[sample] = cnv_seg_df_mod
        cnv_plot_dict[sample] = cnv_plot
        trace_dict[sample] = (start_trace, end_trace)

    seg_trees = get_segment_interval_trees(pd.concat(cnv_seg_dict.values()))
    participant_maf_df = gen_maf(maf_fn, purity_dict, ploidy_dict, seg_trees, maf_start_pos_col, maf_sample_id_col, maf_chromosome_col, maf_cluster_col)

    return participant_maf_df, cnv_plot_dict, cnv_seg_dict, trace_dict


def gen_maf(maf_fn, purity_dict, ploidy_dict, seg_trees, maf_start_pos_col, maf_sample_id_col, maf_chromosome_col, maf_cluster_col):
    """

    Parameters
    ----------
    maf_fn: str
        Filename for the participant maf file
    purity_dict: dict
        Dictionary with purity values for this participant, keys given by sample_id
    ploidy_dict: dict
        Dictionary with ploidy values for this participant, keys given by sample_id
    seg_trees: list of IntervalTrees
        IntervalTree for each contig with copy number data

    Returns
    -------
    maf_df: pd.DataFrame
        Modified maf for this participant, including copy number data and additional annotations
    """
    maf_df = cached_read_csv(maf_fn, sep='\t')
    # start_pos = maf_df.columns[maf_df.columns.isin(['Start_position', 'Start_Position'])][0]
    alt = maf_df.columns[maf_df.columns.isin(['Tumor_Seq_Allele2', 'Tumor_Seq_Allele'])][0]
    # sample_id_col = maf_df.columns[maf_df.columns.isin(['Tumor_Sample_Barcode', 'Sample_ID', 'sample_id', 'Sample_id'])][0]
    maf_df['id'] = maf_df.apply(lambda x: get_unique_identifier(x, start_pos=maf_start_pos_col, alt=alt), axis=1)

    maf_df['Sample_ID'] = maf_df[maf_sample_id_col]

    maf_sample_names = set(maf_df['Sample_ID'])
    given_sample_names = set(purity_dict.keys())
    if len(maf_sample_names & given_sample_names) == 0:
        if len(set([p[:-5] for p in maf_sample_names]) & given_sample_names) == 0:
            raise ValueError("Maf sample names don't match what is given")
        else:
            # remove '_pair' from sample ids in maf
            maf_df['Sample_ID'] = maf_df['Sample_ID'].apply(lambda x: x[:-5])

    # maf_df = switch_contigs(maf_df)
    # .replace giving: Series.replace cannot use dict-like to_replace and non-None value ?? todo
    maf_df[maf_chromosome_col] = maf_df[maf_chromosome_col].apply(lambda x: '23' if x == 'X' else '24' if x == 'Y' else x)

    maf_df = apply_segment_data_to_df(maf_df, seg_trees)
    maf_df.set_index('id', inplace=True, drop=False)

    c_values = [calc_cn_levels(pur, plo) for pur, plo in zip(purity_dict.values(), ploidy_dict.values())]
    c_0 = {sample: c[0] for sample, c in zip(purity_dict.keys(), c_values)}
    c_delta = {sample: c[1] for sample, c in zip(purity_dict.keys(), c_values)}

    maf_df['purity'] = maf_df['Sample_ID'].replace(purity_dict)
    maf_df['ploidy'] = maf_df['Sample_ID'].replace(ploidy_dict)
    maf_df['c_0'] = maf_df['Sample_ID'].replace(c_0)
    maf_df['c_delta'] = maf_df['Sample_ID'].replace(c_delta)

    # cluster_assignment = maf_df.columns[maf_df.columns.isin(['Cluster_Assignment', 'cluster'])][0]
    maf_df[maf_chromosome_col] = maf_df[maf_chromosome_col].astype(int)
    if maf_cluster_col in maf_df:
        maf_df['cluster_color'] = maf_df[maf_cluster_col].astype(int).apply(lambda x: cluster_color(x))

    c_size_cumsum = np.cumsum([0] + list(csize.values()))
    maf_df['x_loc'] = maf_df.apply(lambda x: c_size_cumsum[x[maf_chromosome_col] - 1] + x[maf_start_pos_col], axis=1)
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

    return maf_df

# Adapting https://github.com/getzlab/cnv_suite/blob/1cfef02623e93d7483841b50dfc598dcfebe577e/cnv_suite/visualize/plot_cnv_profile.py#L93C1-L125C15 to plot background on all subfigures
def updated_plot_acr_subplots(fig_list, title, fig_names, csize, height_per_sample=350, **kwargs):
    """Add each Figure in list to plotly subplots.

    Additional kwargs are passed to plotly's make_subplots method.

    :param height_per_sample: plot height for each sample plot (scales based on size of fig_list); default 350
    :param csize: dict with chromosome sizes, as {contig_name: size}
    :param fig_list: List of plotly.graph_objects.Figure, one for each row
    :param title: Title of plot
    :param fig_names: Title for each subplot (one for each row)
    """
    fig = plotly.subplots.make_subplots(rows=len(fig_list), cols=1,
                                        shared_xaxes=True, subplot_titles=fig_names,
                                        vertical_spacing=0.15/len(fig_list), **kwargs)

    for i, sub_figure in enumerate(fig_list):
        fig.add_traces(sub_figure.data, rows=i + 1, cols=1)
        fig.update_yaxes(fig_list[i].layout.yaxis, row=i+1, col=1)
        # Add chromosome background back in
        add_background(fig, csize.keys(), csize, plotly_row=i+1, plotly_col=1)

    # update height based on subplot number
    fig.update_layout(height=len(fig_list)*height_per_sample + 50,
                      title_text=title,
                      title_font_size=26,
                      plot_bgcolor='white')
    fig.update_xaxes(fig_list[0].layout.xaxis)

    # show x-axis title on only bottom plot
    fig.update_xaxes(title_text='', selector=(lambda x: not x['showticklabels']))

    return fig
    
def gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, maf_sample_id_col, maf_start_pos_col, maf_chromosome_col, maf_cluster_col, maf_hugo_col, maf_variant_class_col, maf_protein_change_col):
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
    preprocess_data_dir  TODO: remove
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

    # collect Figures and mutation maf
    purity_dict = samples_df.loc[sample_list, 'wxs_purity'].to_dict()
    ploidy_dict = samples_df.loc[sample_list, 'wxs_ploidy'].to_dict()
    cnv_seg_filenames = samples_df.loc[sample_list, 'cnv_seg_fn'].values.tolist()
    maf_fn = df.loc[idx, 'maf_fn']
    participant_maf_df, cnv_plot_dict, cnv_seg_dict, trace_dict = gen_participant_cnv_and_maf(
        cnv_seg_filenames, 
        maf_fn, 
        sample_list, 
        csize, 
        purity_dict, 
        ploidy_dict, 
        maf_start_pos_col, 
        maf_sample_id_col, 
        maf_chromosome_col, 
        maf_cluster_col
    )

    fig_list = []
    for sample_id in sample_selection_corrected:
        cnv_plot = cnv_plot_dict[sample_id]
        this_seg_df = cnv_seg_dict[sample_id]
        update_cnv_color_absolute(cnv_plot, this_seg_df, absolute_val, color,
                                  trace_dict[sample_id][0], trace_dict[sample_id][1])
        if absolute_val:
            cnv_plot.update_yaxes(title_text="Absolute Copy Number")
        else:
            cnv_plot.update_yaxes(title_text="Allelic Copy Ratio")
        fig_list.append(cnv_plot)

    cnv_subplots_fig = updated_plot_acr_subplots(fig_list, 'Copy Number Plots', sample_selection_corrected, csize)
    
    update_cnv_scatter_sigma_toggle(cnv_subplots_fig, sigmas_val)
    
    if selected_mutation_rows:
        participant_maf_df = participant_maf_df.loc[selected_mutation_rows]
    elif filtered_mutation_rows:
        participant_maf_df = participant_maf_df.loc[filtered_mutation_rows]
    # else (if all mutations in table are filtered out and none selected): use all mutations

    for i, sample_id in enumerate(sample_selection_corrected):
        sample_maf_df = participant_maf_df[participant_maf_df[maf_sample_id_col] == sample_id]
        cnv_subplots_fig.add_trace(
            gen_mut_scatter(
                sample_maf_df, 
                sigmas_val, 
                sample_id, 
                maf_cluster_col, 
                maf_hugo_col, 
                maf_chromosome_col, 
                maf_start_pos_col, 
                maf_variant_class_col, 
                maf_protein_change_col
            ), 
            row=i+1, col=1
        )

    return [
        cnv_subplots_fig,
        sample_list,
        sample_selection_corrected
    ]

def gen_absolute_components(
    data: PatientSampleData, 
    idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, selected_mutation_rows, filtered_mutation_rows, 
    maf_sample_id_col=None, maf_start_pos_col=None, maf_chromosome_col=None, maf_cluster_col=None, maf_hugo_col=None, maf_variant_class_col=None, 
    maf_protein_change_col=None):
    """Generate CNV plot - new data callback.
    
    Parameters
    ----------
    data
        PatientSampleData containing participant and sample dfs
    idx
        Index - current participant
    sample_selection
        Samples selected in checkboxes to be displayed 
    sigmas
        Sigma checkbox value (checked or unchecked)
    color
        Color seleceted in checkboxes to be displayed 
    absolute
        Display Absolute CN checkbox value (checked or unchecked)
    button_clicks 
        Submit button response 
    cnv_plot
    sample_list 
    selected_mutation_rows 
        Rows selected in the mutation table 
    filtered_mutation_rows 
        Rows displayed in the mutation table after filtering 
    maf_sample_id_col
        kwarg - Name of the sample id column in the maf file 
    maf_start_pos_col
        kwarg - Name of the start position column in the maf file
    maf_chromosome_col
        kwarg - Name of the chromosome column in the maf file
    maf_cluster_col
        kwarg - Name of the cluster assignment column in the maf file  
    maf_hugo_col
        kwarg - Name of the hugo symbol column in the maf file 
    maf_variant_class_col
        kwarg - Name of the variant classification column in the maf file 
    maf_protein_change_col
        kwarg - Name of the protein change column in the maf file 

    Returns
    -------
    Corresponding values to callback outputs (in corresponding order)

    """
    df = data.participant_df
    samples_df = data.sample_df

    # when changing participants, show all mutations at first
    # the filtered and selected mutations input to this function are from the old (previous participant's) MutationTable
    filtered_mutation_rows = None
    selected_mutation_rows = None

    cnv_plot, sample_list, sample_selection = gen_cnv_plot(df, idx, [], sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, maf_sample_id_col, maf_start_pos_col, maf_chromosome_col, maf_cluster_col, maf_hugo_col, maf_variant_class_col, maf_protein_change_col)
    button_clicks = None

    return [
        cnv_plot,
        sample_list,
        sample_selection,
        button_clicks
    ]

def internal_gen_absolute_components(data: PatientSampleData, idx, sample_selection, sigmas, color, absolute, button_clicks, cnv_plot, sample_list, selected_mutation_rows, filtered_mutation_rows, maf_sample_id_col=None, maf_start_pos_col=None, maf_chromosome_col=None, maf_cluster_col=None, maf_hugo_col=None, maf_variant_class_col=None, maf_protein_change_col=None):
    """Generate CNV plot - internal callback.
    
    Parameters
    ----------
    data
        PatientSampleData containing participant and sample dfs
    idx
        Index - current participant
    sample_selection
        Samples selected in checkboxes to be displayed 
    sigmas
        Sigma checkbox value (checked or unchecked)
    color
        Color seleceted in checkboxes to be displayed 
    absolute
        Display Absolute CN checkbox value (checked or unchecked)
    button_clicks 
        Submit button response 
    cnv_plot
    sample_list 
    selected_mutation_rows 
        Rows selected in the mutation table 
    filtered_mutation_rows 
        Rows displayed in the mutation table after filtering 
    maf_sample_id_col
        kwarg - Name of the sample id column in the maf file 
    maf_start_pos_col
        kwarg - Name of the start position column in the maf file
    maf_chromosome_col
        kwarg - Name of the chromosome column in the maf file
    maf_cluster_col
        kwarg - Name of the cluster assignment column in the maf file  
    maf_hugo_col
        kwarg - Name of the hugo symbol column in the maf file 
    maf_variant_class_col
        kwarg - Name of the variant classification column in the maf file 
    maf_protein_change_col
        kwarg - Name of the protein change column in the maf file 

    Returns
    -------
    Corresponding values to callback outputs (in corresponding order)

    """
    df = data.participant_df
    samples_df = data.sample_df

    if button_clicks != None:
        cnv_plot, sample_list, sample_selection = gen_cnv_plot(df, idx, sample_selection, sigmas, color, absolute, selected_mutation_rows, filtered_mutation_rows, samples_df, maf_sample_id_col, maf_start_pos_col, maf_chromosome_col, maf_cluster_col, maf_hugo_col, maf_variant_class_col, maf_protein_change_col)
        button_clicks = None

    return [
        cnv_plot,
        sample_list,
        sample_selection,
        button_clicks
    ]
