import pandas as pd
from typing import List, Dict
import os
import pickle

from JupyterReviewer.ReviewData import ReviewData
from JupyterReviewer.Data import DataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
from JupyterReviewer.DataTypes.PatientSampleData import PatientSampleData

from JupyterReviewer.AppComponents.PhylogicComponents import gen_ccf_pmf_component, gen_phylogic_app_component, gen_cluster_metrics_component
from JupyterReviewer.AppComponents.MutationTableComponent import gen_mutation_table_app_component
from JupyterReviewer.AppComponents.CNVPlotComponent import gen_cnv_plot_app_component, gen_preloaded_cnv_plot


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

    def gen_data(self,
                 description: str,
                 participant_df: pd.DataFrame,
                 sample_df: pd.DataFrame,
                 preprocess_data_dir: str,
                 annot_df: pd.DataFrame = None,
                 annot_col_config_dict: Dict = None,
                 history_df: pd.DataFrame = None,
                 index: List = None,
                 reload_cnv_figs=False,
                 ) -> PatientSampleData:
        """

        Parameters
        ----------
        description
            Describe the review session. This is useful if you copy the history of this object to a new review data
            object
        participant_df
            dataframe containing participant data. this will be the primary dataframe
        sample_df
            dataframe containing sample data
        annot_df
            Dataframe with previous/prefilled annotations
        annot_col_config_dict
            Dictionary specifying active annotation columns and validation configurations
        history_df
            Dataframe with previous/prefilled history

        Returns
        -------
        PatientSampleData
            A `Data` object for phylogic review and annotation history

        """
        if index is None:
            index = participant_df.index.tolist()

        # preprocessing todo
        cnv_figs_dir = os.path.join(preprocess_data_dir, 'cnv_figs')
        if not os.path.exists(cnv_figs_dir):
            os.makedirs(cnv_figs_dir)
            reload_cnv_figs = True
        else:
            print(f'cnv figs directory already exists: {cnv_figs_dir}')

        if reload_cnv_figs:
            for sample in sample_df.index.tolist():
                output_fn = f'{cnv_figs_dir}/{sample}.cnv_fig.pkl'
                fig, start_trace, end_trace = gen_preloaded_cnv_plot(sample_df, sample)
                pickle.dump([fig, start_trace, end_trace], open(output_fn, 'wb'))

        # todo get number of tree options for tree validation; don't think this is possible now

        # create review data object
        rd = PatientSampleData(index=index, description=description,
                               participant_df=participant_df, sample_df=sample_df,
                               annot_df=annot_df, annot_col_config_dict=annot_col_config_dict, history_df=history_df)
        return rd

    def set_default_review_data_annotations(self):
        self.add_review_data_annotation('cluster_annotation', DataAnnotation('string'))
        self.add_review_data_annotation('selected_tree_idx', DataAnnotation('int', default=1))  # options=range(1, tree_num+1) how to access this?
        self.add_review_data_annotation('selected_tree', DataAnnotation('string'))
        self.add_review_data_annotation('notes', DataAnnotation('string'))
        # 'variant_blocklist': ReviewDataAnnotation(),  # needs to go in separate reviewer

    def set_default_review_data_annotations_app_display(self):
        self.add_review_data_annotations_app_display('cluster_annotation', 'text')
        self.add_review_data_annotations_app_display('selected_tree_idx', 'number')
        self.add_review_data_annotations_app_display('selected_tree', 'text')
        self.add_review_data_annotations_app_display('notes', 'textarea')

    def gen_review_app(self, preprocess_data_dir, custom_colors=[], drivers_fn=None) -> ReviewDataApp:  #todo change empty list to None
        """Generates a ReviewDataApp object

        Parameters
        ----------
        preprocess_data_dir
            Directory to store pre-processed data (like cnv pickle files)
        custom_colors
            List of custom colors (for what?)
        drivers_fn
            Path and filename for driver genes; should be single column with no header

        Returns
        -------
        ReviewDataApp
        """
        app = ReviewDataApp()

        app.add_component(gen_mutation_table_app_component(), custom_colors=custom_colors)
        app.add_component(gen_phylogic_app_component(), drivers_fn=drivers_fn)
        app.add_component(gen_cluster_metrics_component())
        app.add_component(gen_cnv_plot_app_component(), preprocess_data_dir=preprocess_data_dir)
        app.add_component(gen_ccf_pmf_component())

        return app

    def set_default_autofill(self):
        pass
