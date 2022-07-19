import pandas as pd

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate

from JupyterReviewer.AppComponents.PhylogicComponents import gen_ccf_pmf_component, gen_phylogic_app_component
from JupyterReviewer.AppComponents.MutationTableComponent import gen_mutation_table_app_component
from JupyterReviewer.AppComponents.CNVPlotComponent import gen_cnv_plot_app_component


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

    def gen_review_app(self, sample_data_df, preprocess_data_dir, custom_colors=[], drivers_fn=None) -> ReviewDataApp:  #todo change empty list to None
        app = ReviewDataApp()

        app.add_component(gen_mutation_table_app_component(), custom_colors=custom_colors)

        app.add_component(gen_phylogic_app_component(), drivers_fn=drivers_fn, biospecimens_fn=sample_data_df)

        app.add_component(gen_cnv_plot_app_component(), samples_fn=sample_data_df, preprocess_data_dir=preprocess_data_dir)

        # app.add_component(cluster_metrics)

        # todo finish mutation table updates
        # todo add button to switch mutations
        # do we want to allow multiple types of mutation selection (filter vs. selection)?
        app.add_component(gen_ccf_pmf_component())

        return app

    def gen_autofill(self):
        pass
