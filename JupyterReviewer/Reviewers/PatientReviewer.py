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
import dalmatian

from JupyterReviewer.ReviewData import ReviewData, ReviewDataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
from JupyterReviewer.AppComponents.ReviewerLayout import gen_phylogic_components_layout, gen_cnv_plot_layout
from JupyterReviewer.AppComponents.MutationTableComponent import gen_mutation_table_app_component
from JupyterReviewer.AppComponents.PhylogicComponents import gen_phylogic_app_component
from JupyterReviewer.AppComponents.CNVPlotComponent import gen_cnv_plot_app_component


def validate_string_list(x):
    if type(x) == str:
        split_list = [i.strip().isdigit() for i in x.split(',')]
        return all(split_list)
    else:
        return False


def gen_clinical_data_table(df, idx, cols):
    r=df.loc[idx]
    return [dbc.Table.from_dataframe(r[cols].to_frame().reset_index())]

def collect_data(workspace, treatments_fn, treatment_files_path, participant_maf_files_path, participant_file_path, samples_file_path):
    wm = dalmatian.WorkspaceManager(workspace)

    samples_df = wm.get_samples()
    samples_df = samples_df[['participant', 'pdb_collection_date_dfd', 'unmatched_alleliccapseg_tsv', 'unmatched_mutation_validator_validated_maf']].dropna()

    pairs_df = wm.get_pairs()
    pairs_df = pairs_df[['participant', 'alleliccapseg_tsv', 'mutation_validator_validated_maf']].dropna()

    new_samples_df = pd.concat([pairs_df, samples_df])
    new_samples_df = samples_df.reset_index().apply(lambda x: x.str.replace('_pair', ''))
    new_samples_df.rename(columns={'index': 'sample_id'}, inplace=True)
    new_samples_df['cnv_seg_fn'] = new_samples_df.apply(lambda x: x['alleliccapseg_tsv'] if pd.isna(x['unmatched_alleliccapseg_tsv']) else x['unmatched_alleliccapseg_tsv'], axis=1)
    new_samples_df['maf_fn'] = new_samples_df.apply(lambda x: x['mutation_validator_validated_maf'] if pd.isna(x['unmatched_mutation_validator_validated_maf']) else x['unmatched_mutation_validator_validated_maf'], axis=1)
    new_samples_df.dropna(axis=1, inplace=True)

    participants_df = wm.get_participants()
    participants_df = participants_df[['pdb_age_at_diagnosis', 'pdb_death_date_dfd', 'pdb_gender', 'pdb_vital_status']].dropna()

    treatments_df = pd.read_csv(treatments_fn, sep='\t')

    if not os.path.exists(treatment_files_path):
        os.makedirs(treatment_files_path)
    if not os.path.exists(participant_maf_files_path):
        os.makedirs(participant_maf_files_path)

    for participant in participants_df.index:
        treatments_file_name = f'{treatment_files_path}/{participant}_treatment.txt'
        if not os.path.exists(treatments_file_name):
            this_p_treatments = treatments_df[treatments_df['participant_id'] == participant]
            if this_p_treatments.shape[0] > 0:
                this_p_treatments.to_csv(treatments_file_name, sep='\t', index=False)
        participants_df.loc[participant, 'treatments_fn'] = os.path.normpath(treatments_file_name)

        maf_file_name = f'{participant_maf_files_path}/{participant}_maf.txt'
        if not os.path.exists(maf_file_name):
            this_p_mafs = new_samples_df[new_samples_df['participant'] == participant]['maf_fn'].tolist()
            if len(this_p_mafs) > 0:
                this_p_maf = pd.concat([pd.read_csv(maf, sep='\t', encoding = "ISO-8859-1") for maf in this_p_mafs])
                this_p_maf.to_csv(maf_file_name, sep='\t', index=False)
        participants_df.loc[participant, 'maf_fn'] = os.path.normpath(maf_file_name)

    participants_df.dropna(inplace=True)

    if not os.path.exists(participant_file_path):
        os.makedirs(participant_file_path)
    if not os.path.exists(samples_file_path):
        os.makedirs(samples_file_path)

    participant_file_name = f'{participant_file_path}/participants.txt'
    samples_file_name = f'{samples_file_path}/samples.txt'

    if not os.path.exists(participant_file_name):
        participants_df.to_csv(participant_file_name, sep='\t', index=False)
    if not os.path.exists(samples_file_name):
        new_samples_df.to_csv(samples_file_name, sep='\t', index=True)

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
            samples_df = pd.read_csv(samples_fn, sep='\t').dropna()
            samples_df.set_index('sample_id', inplace=True)
            sample_list = samples_df.index.tolist()

            for sample in sample_list:
                output_fn = f'{cnv_figs_dir}/{sample}.cnv_fig.pkl'
                fig = gen_preloaded_cnv_plot(df, samples_df, sample)
                pickle.dump(fig, open(output_fn, 'wb'))
                #samples_df.loc[sample, 'cnv_plots_pkl'] = output_fn

        rd = ReviewData(
            review_data_fn=review_data_fn,
            description=description,
            df=df,
        )

        return rd

    def set_default_review_data_annotations(self):
        self.add_review_data_annotation('Resistance Explained', ReviewDataAnnotation('string',
                                                                                     options=['Mutation', 'CNV',
                                                                                              'Partial/Hypothesized',
                                                                                              'Unknown']))
        self.add_review_data_annotation('Resistance Notes', ReviewDataAnnotation('string'))
        self.add_review_data_annotation('Growing Clones',
                                        ReviewDataAnnotation('string', validate_input=validate_string_list))
        self.add_review_data_annotation('Shrinking Clones',
                                        ReviewDataAnnotation('string', validate_input=validate_string_list))
        self.add_review_data_annotation('Annotations', ReviewDataAnnotation('string', options=['Hypermutated',
                                                                                               'Convergent Evolution',
                                                                                               'Strong clonal changes']))
        self.add_review_data_annotation('Selected Tree (idx)', ReviewDataAnnotation('int', default=1))
        self.add_review_data_annotation('Other Notes', ReviewDataAnnotation('string'))

    def set_default_review_data_annotations_app_display(self):
        self.add_review_data_annotations_app_display('Resistance Explained', 'radioitem')
        self.add_review_data_annotations_app_display('Resistance Notes', 'textarea')
        self.add_review_data_annotations_app_display('Growing Clones', 'text')
        self.add_review_data_annotations_app_display('Shrinking Clones', 'text')
        self.add_review_data_annotations_app_display('Annotations', 'checklist')
        self.add_review_data_annotations_app_display('Selected Tree (idx)', 'number')
        self.add_review_data_annotations_app_display('Other Notes', 'textarea')

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

        return app

    def set_default_autofill(self):
        self.add_autofill('Phylogic Graphics', State('tree-dropdown', 'value'), 'Selected Tree (idx)')
