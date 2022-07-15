import cnv_suite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from scipy.stats import beta
import re

from cnv_suite.visualize import plot_acr_interactive, update_cnv_scatter_cn, update_cnv_scatter_color, update_cnv_scatter_sigma_toggle
from cnv_suite import calc_avg_cn, calc_absolute_cn, calc_cn_levels, return_seg_data_at_loci, apply_segment_data_to_df, get_segment_interval_trees, switch_contigs

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate


class PhylogicReviewer(ReviewerTemplate):
    """Class to facilitate reviewing Phylogic results in a consistent and efficient manner.

    Notes
    -----
    - Display CCF plot (pull code from Patient Reviewer)
    - Display mutations (pull code from Patient Reviewer)
    - Display trees (pull code from Patient Reviewer?)
        - Work on weighted tree viz
    - Display CN plots (pull code from old phylogic review code, with updated library use)**
         - Display mutations on CN plots
         - Allow filtering using mutations table
    - Metrics for coding vs. non_coding, silent vs. non_syn (coding), indels vs. SNPs summarized for each cluster**
        - Could output some statistics as well -> highlight sig differences from null
    - Integrated (link to run bash script) variant reviewer
    - View CCF pmf distributions for individual mutations**

    - Annotate:
        - Variants (add to variant_review file)
        - Cluster annotations (add to cluster blocklist? Only for final analysis though - better to block mutations)
            - Can also just annotate with notes
        - Notes generally
        - Select correct tree (save tree child-parent relationship, but how to associate clusters if you re-run?)
    """

    def gen_review_data(self,
                        review_data_fn: str,
                        description: str='',
                        df: pd.DataFrame = pd.DataFrame(),
                        reuse_existing_review_data_fn: str = None, *args, **kwargs) -> ReviewData:
        """

        Parameters
        ----------
        review_data_fn
        description
            Description of this phylogic data review session
        df
            dataframe containing input data, including columns
            (mut_ccfs, cluster_ccfs, tree_tsv, unclustered_muts, cell_pop_mcmc_trace).
        reuse_existing_review_data_fn

        Returns
        -------
        ReviewData
            A `ReviewData` object for phylogic review

        """

        # preprocessing todo
        # todo get number of tree options for tree validation; don't think this is possible now

        # create review data object
        rd = ReviewData(review_data_fn=review_data_fn,
                        description=description,
                        df=df,
                        reuse_existing_review_data_fn=reuse_existing_review_data_fn)
        return rd

    def gen_review_data_annotations(self):
        self.add_review_data_annotation('cluster_annotation', ReviewDataAnnotation('string'))
        self.add_review_data_annotation('selected_tree_idx', ReviewDataAnnotation('int', default=1))  # options=range(1, tree_num+1) how to access this?
        self.add_review_data_annotation('selected_tree', ReviewDataAnnotation('string'))
        self.add_review_data_annotation('notes', ReviewDataAnnotation('string'))
        # 'variant_blocklist': ReviewDataAnnotation(),  # needs to go in separate reviewer

    def gen_review_data_annotations_app_display(self):
        self.add_review_data_annotations_app_display('cluster_annotation', 'text')
        self.add_review_data_annotations_app_display('selected_tree_idx', 'number')
        self.add_review_data_annotations_app_display('selected_tree', 'text')
        self.add_review_data_annotations_app_display('notes', 'textarea')

    def gen_review_app(self, sample_data_df) -> ReviewDataApp:
        app = ReviewDataApp()

        # app.add_component(ccf_plot)
        # app.add_component(tree)
        # app.add_component(mutation_table)
        app.add_component(AppComponent(name='CNV plot with mutations',
                                       layout=html.Div([
                                           html.Div([
                                               dcc.RadioItems(
                                                   ['Differential', 'Cluster', 'Red/Blue', 'Clonal/Subclonal'],
                                                   'Differential',
                                                   id='cnv-color-radio-item'
                                               ),
                                               dcc.Checklist(options={'cnv_sigma': 'Show CNV sigmas'},
                                                             value=['cnv_sigma'],
                                                             id='cnv-sigma-box'
                                                             ),
                                               dcc.Checklist(options={'mut_sigma': 'Show Mutation uncertainty'},
                                                             value=['mut_sigma'],
                                                             id='mut-sigma-box'
                                                             ),
                                               dcc.RadioItems(
                                                   ['Cluster', 'Multiplicity', 'Clonal_Subclonal'],  # add option to scale y-axis by multiplicity (rather than scaled AF?)
                                                   'Cluster',
                                                   id='mut-color-radio-item'
                                               ),
                                               dcc.Checklist(options={'zero_vaf': 'Show Zero-VAF mutations'},
                                                             value=['zero_vaf'],
                                                             id='zero-vaf-box'
                                                             ),
                                               dcc.Checklist(options={'absolute_cn': 'Display Absolute CN'},
                                                             value=['absolute_cn'],
                                                             id='absolute-cnv-box'
                                                             ),
                                               dcc.Checklist(options=[],
                                                             value=[],
                                                             id='sample-selection')
                                           ]),
                                           # filter using mutation table**

                                           dcc.Graph(id='cnv-plot',
                                                     figure=go.Figure()
                                                     ),
                                       ]),
                                       callback_input=[
                                           Input('cnv-color-radio-item', 'value'),
                                           Input('cnv-sigma-box', 'value'),
                                           Input('mut-sigma-box', 'value'),
                                           Input('mut-color-radio-item', 'value'),
                                           Input('zero-vaf-box', 'value'),
                                           Input('absolute-cnv-box', 'value'),
                                           Input('sample-selection', 'value'),
                                       ],
                                       callback_output=[
                                           Output('cnv-plot', 'figure'),
                                           Output('sample-selection', 'options'),
                                           Output('sample-selection', 'value')
                                       ],
                                       callback_state=[
                                           State('mutation-table', 'selected_row_ids')
                                       ],

                                       new_data_callback=cnv_plot_w_mutations,
                                       internal_callback=cnv_plot_w_mutations,

                                       # callback_state=[] todo pull from mutation table?
                                       ),
                                       sample_data_df=sample_data_df
        )

        # app.add_component(cluster_metrics)

        # app.add_component(ccf_pmf_mutation_plot)

        return app

    def gen_autofill(self):
        pass


def cnv_plot_w_mutations(data_df, idx, cnv_color, cnv_sigma, mut_sigma, mut_color, zero_vaf, absolute_cnv, sample_selection, sample_data_df):
    csize = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431,
              135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,
              63025520, 48129895, 51304566, 156040895, 57227415]

    sample_list = sample_data_df[sample_data_df['participant_id'] == idx].sort_values('collection_date_dfd').index.tolist()
    # restrict sample selection to only two samples at a time
    sample_selection_corrected = [sample_list[0]] if sample_selection == [] else sample_selection[:2]

    mut_ccfs_df = switch_contigs(pd.read_csv(data_df.loc[idx]['mut_ccfs'], sep='\t'))
    mut_ccfs_df['Chromosome'] = mut_ccfs_df['Chromosome'].astype(int)  # todo change
    seg_df = []
    for sample_id in sample_selection_corrected:
        this_seg_df = pd.read_csv(sample_data_df.loc[sample_id, 'alleliccapseg_tsv'], sep='\t')
        this_seg_df['Sample_ID'] = sample_id
        seg_df.append(this_seg_df)

    if 'seg_cluster' in sample_data_df:
        seg_cluster_df = []
        for sample_id in sample_selection_corrected:
            this_cluster_df = pd.read_csv(sample_data_df.loc[sample_id, 'seg_cluster'], sep='\t')
            this_cluster_df['Sample_ID'] = sample_id
            seg_cluster_df.append(this_seg_df)

        seg_trees = get_segment_interval_trees(pd.concat(seg_df), pd.concat(seg_cluster_df))
    else:
        seg_cluster_df = None
        seg_trees = get_segment_interval_trees(pd.concat(seg_df))

    mut_ccfs_df = apply_segment_data_to_df(mut_ccfs_df, seg_trees)
    phylogic_color_dict = get_phylogic_color_scale()

    c_size_cumsum = np.cumsum([0] + csize)
    mut_ccfs_df['x_loc'] = mut_ccfs_df.apply(lambda x: c_size_cumsum[x['Chromosome'] - 1] + x['Start_position'], axis=1)
    mut_ccfs_df['cluster_color'] = mut_ccfs_df['Cluster_Assignment'].apply(lambda x: phylogic_color_dict[x])
    mut_ccfs_df['VAF'] = mut_ccfs_df['t_alt_count'] / (mut_ccfs_df['t_alt_count'] + mut_ccfs_df['t_ref_count'])

    cnv_plot = make_subplots(1, len(sample_selection_corrected))

    for i, sample_id in enumerate(sample_selection_corrected):
        trace_start, trace_end = plot_acr_interactive(seg_df[sample_id], cnv_plot, csize=csize, segment_colors=cnv_color, sigmas=cnv_sigma, row=i+1)
        purity = sample_data_df.loc[sample_id, 'wxs_purity']
        ploidy = sample_data_df.loc[sample_id, 'wxs_ploidy']
        c_0, c_delta = calc_cn_levels(purity, ploidy)

        this_mut_df = mut_ccfs_df[mut_ccfs_df['Sample_ID'] == sample_id]
        this_mut_df['mu_major_adj'] = (this_mut_df['mu_major'] - c_0) / c_delta
        this_mut_df['mu_minor_adj'] = (this_mut_df['mu_minor'] - c_0) / c_delta

        this_mut_df['multiplicity_ccf'] = this_mut_df.apply(
            lambda x: x.VAF * (purity * (x.mu_major_adj + x.mu_minor_adj) +
                               2 * (1 - purity)) / purity, axis=1)

        # calculate error bars for mutations
        this_mut_df['error_top'] = this_mut_df.apply(
            lambda x: calculate_error(x.t_alt_count, x.t_ref_count, purity, 0.975), axis=1)
        this_mut_df['error_bottom'] = this_mut_df.apply(
            lambda x: -1 * calculate_error(x.t_alt_count, x.t_ref_count, purity, 0.025), axis=1)

        cnv_plot.add_trace(make_mut_scatter(this_mut_df, mut_sigma, mut_color, zero_vaf), row=str(i+1))

    return [cnv_plot,
            sample_list,
            sample_selection_corrected]


def make_mut_scatter(mut_df, mut_sigma, mut_color, zero_vaf):
    """Create a scatter plot with all mutations in the dataframe.

    Not using plotly express because it returns a separate trace for each color (each cluster).

    TO-DO
    -----
    Allow for selecting/filtering mutations with table
    """
    if not zero_vaf:
        mut_df = mut_df[mut_df['VAF'] > 0]

    mut_scatter = go.Scatter(x=mut_df['x_loc'], y=mut_df['multiplicity_ccf'],
                               mode='markers', marker_size=10,
                               marker_color=mut_df['cluster_color'],
                               error_y=dict(type='data',
                                            array=mut_df['error_top'],
                                            arrayminus=mut_df['error_bottom'],
                                            color='gray',
                                            visible=mut_sigma,
                                            width=0),
                               customdata=np.stack((mut_df['Hugo_Symbol'].tolist(),
                                                    mut_df['Chromosome'].tolist(),
                                                    mut_df['Start_position'].tolist(),
                                                    mut_df['VAF'].tolist(),
                                                    mut_df['Cluster_Assignment'].tolist(),
                                                    mut_df['Variant_Type'].tolist(),
                                                    mut_df['Variant_Classification'].tolist(),
                                                    mut_df['Protein_change']),
                                                   axis=-1),
                               hovertemplate='<extra></extra>' +
                                             'Gene: %{customdata[0]} %{customdata[1]}:%{customdata[2]} <br>' +
                                             'Variant: %{customdata[5]}, %{customdata[6]} <br>' +
                                             'Protein Change: %{customdata[7]} <br>' +
                                             'Multiplicity: %{y:.3f} <br>' +
                                             'VAF: %{customdata[3]:.3f} <br>' +
                                             'Cluster: %{customdata[4]:d}')
    return mut_scatter


def calculate_error(alt, ref, purity, percentile):
    if alt == 0:
        return 0
    else:
        return (beta.ppf(percentile, alt, ref) - alt / (alt + ref)) / purity


def ccf_pmf_plot(data_df, idx, mut_ids, sample_selection):
    """Plots the CCF pmf distribution for the chosen mutation(s).

    Notes
    -----
    - Displays the pmf distribution as a normalized histogram
    - Samples are shown in separate rows
    - Clusters displayed with different colors, with adjacent bars

    """
    mut_ccfs_df = pd.read_csv(data_df.loc[idx, 'mut_ccfs'], sep='\t')
    mut_ccfs_df['unique_mut_id'] = mut_ccfs_df.apply(get_unique_identifier, axis=1)
    chosen_muts_df = mut_ccfs_df[mut_ccfs_df['unique_mut_id'].isin(mut_ids)].copy()

    ccfs_headers = [re.search('.*[01].[0-9]+', i) for i in mut_ccfs_df.columns]
    ccfs_headers = [x.group() for x in ccfs_headers if x]

    #pmf_plot = make_subplots(1, len(sample_selection))
    stacked_muts = chosen_muts_df.set_index(['Sample_ID', 'unique_mut_id', 'Cluster_Assignment'])[
        ccfs_headers].stack().reset_index().rename(columns={'level_3': 'ccf_val', 0: 'pmf_val'})
    fig = px.histogram(stacked_muts, x='ccf_val', y='pmf_val', color='Cluster_Assignment', facet_row='Sample_ID', barmode='group', histfunc='avg')
    return fig


def get_unique_identifier(row, chrom='Chromosome', start_pos='Start_position',
                          ref='Reference_Allele', alt='Tumor_Seq_Allele'):
    """Generates unique string for this mutation, including contig, start position, ref and alt alleles.

    Does not include End Position, for this field is not present in mut_ccfs Phylogic output. However, specification of both the alt and ref alleles are enough to distinguish InDels.

    :param row: pd.Series giving the data for one mutation from a maf or maf-like dataframe
    :param chrom: the name of the contig/chromosome column/field; default: Chromosome
    :param start_pos: the name of the start position column/field; default: Start_position
    :param ref: the name of the reference allele column/field; default: Reference_Allele
    :param alt: the name of the alternate allele column/field; default: Tumor_Seq_Allele
    """
    return f"{row[chrom]}:{row[start_pos]}{row[ref]}>{row[alt]}"


def get_phylogic_color_scale():
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
    return colors_dict


def get_hex_string(c):
    return '#{:02X}{:02X}{:02X}'.format(*c)
