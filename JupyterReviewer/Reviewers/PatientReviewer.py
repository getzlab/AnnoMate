"""PatientReviewer.py module

Interactive dashboard for reviewing and annotating data on a patient-by-patient basis
Includes app layout and callback functionality

Run by the user with a Jupyter Notbook: UserPatientReviewer.ipynb

"""

import pandas as pd
import pickle
import os
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
from JupyterReviewer.AppComponents.ReviewerLayout import gen_phylogic_components_layout, gen_cnv_plot_layout
from JupyterReviewer.AppComponents.MutationTableComponent import gen_mutation_table_app_component
from JupyterReviewer.AppComponents.PhylogicComponents import gen_phylogic_app_component
from JupyterReviewer.AppComponents.CNVPlotComponent import gen_cnv_plot_app_component

def validate_purity(x):
    return (x >= 0) and (x <= 1)

def validate_ploidy(x):
    return x >= 0

# clinical data table functions
def gen_clinical_data_table(df, idx, cols):
    r=df.loc[idx]
    return [dbc.Table.from_dataframe(r[cols].to_frame().reset_index())]

class PatientReviewer(ReviewerTemplate):
    """Interactively review multiple types of data on a patient-by-patient basis.

    Notes
    -----
    - Display clinical data
    - Display cusomizable mutation table
    - Display CCF plot
    - Display Phylogic trees
    - Display cusomizable CNV plot

    """

    def gen_review_data(
        self,
        review_data_fn: str,
        description: str='',
        df: pd.DataFrame = pd.DataFrame(),
        review_data_annotation_dict: {str: ReviewDataAnnotation} = {},
        reuse_existing_review_data_fn: str = None,
        preprocess_data_dir='.',
        reload_cnv_figs=False,
        samples_fn=''
    ):

        """

        Parameters
        ----------
        review_data_fn
            name of a pkl file path where review data is stored
        description
            description of the data source and purpose
        df
            dataframe containing the data to be reviewed
            contains build_tree_posterior_fn, cluster_ccfs_fn, maf_fn, treatments_fn
        review_data_annotation_dict
            dictionary containing annotation fields
        reuse_existing_review_data_fn
        preprocess_data_dir
            directory for preproccessing data to be stored

        Returns
        -------
        ReviewData object

        """

        cnv_figs_dir = f'{preprocess_data_dir}/cnv_fig'
        if not os.path.exists(cnv_figs_dir):
            os.makedirs(cnv_figs_dir)
            reload_cnv_figs = True
        else:
            print(f'cnv figs directory already exists: {cnv_figs_dir}')

        if reload_cnv_figs:
            samples_df = pd.read_csv(samples_fn).dropna()
            samples_df.set_index('Sample_ID', inplace=True)
            sample_list = samples_df.index.tolist()

            for sample in sample_list:
                output_fn = f'{cnv_figs_dir}/{sample}.cnv_fig.pkl'
                fig = gen_preloaded_cnv_plot(df, samples_df, sample)
                pickle.dump(fig, open(output_fn, 'wb'))
                #samples_df.loc[sample, 'cnv_plots_pkl'] = output_fn

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
    def gen_review_app(self, biospecimens_fn, samples_fn, preprocess_data_dir, custom_colors=[], drivers_fn=None) -> ReviewDataApp:
        """Generate app layout.

        Parameters
        ----------
        biospecimens_df
            dataframe containing biospecimens data from the Cancer Drug Resistance Portal
        custom_colors : list of lists
            specify colummn colors in mutation table with format:
            [[column_id_1, filter_query_1, text_color_1, background_color_1]]
        drivers_fn
            file path to csv file of drivers
        samples_fn
            file path to csv file of sample information including cnv_seg_fn (alleliccapseg_tsv)

        Returns
        -------
        ReviewDataApp object

        """

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

        app.add_component(gen_mutation_table_app_component(), custom_colors=custom_colors)

        app.add_component(gen_phylogic_app_component(), drivers_fn=drivers_fn, biospecimens_fn=biospecimens_fn)

        app.add_component(gen_cnv_plot_app_component(), samples_fn=samples_fn, preprocess_data_dir=preprocess_data_dir)

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
