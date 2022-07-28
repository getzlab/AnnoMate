"""PatientReviewer.py module

Interactive dashboard for reviewing and annotating data on a patient-by-patient basis
Includes app layout and callback functionality

Run by the user with a Jupyter Notbook: UserPatientReviewer.ipynb

"""

import pandas as pd
import numpy as np
import pickle
import os
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dalmatian
from typing import Dict, List
import yaml

from JupyterReviewer.ReviewData import ReviewData
from JupyterReviewer.Data import DataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
from JupyterReviewer.AppComponents.MutationTableComponent import gen_mutation_table_app_component
from JupyterReviewer.AppComponents.PhylogicComponents import gen_phylogic_app_component
from JupyterReviewer.AppComponents.CNVPlotComponent import gen_cnv_plot_app_component, gen_preloaded_cnv_plot
from JupyterReviewer.DataTypes.PatientSampleData import PatientSampleData

def validate_string_list(x):
    if type(x) == str:
        split_list = [i.strip().isdigit() for i in x.split(',')]
        return all(split_list)
    else:
        return False

def check_required_inputs(required_inputs):
    for input in required_inputs:
        if required_inputs[input] == None:
            raise ValueError(f'Required input was not given: {input}')

def parse_patient_reviewer_input(config_path):
    # load yaml file
    with open(config_path, 'r') as file:
        try:
            config_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # check for required inputs
    required_inputs = {
        'workspace': config_dict['workspace'],
        'data_path': config_dict['data_path'],
        'clinical_file file_name': config_dict['clinical_file']['file_name'],
        'maf_files table_origin': config_dict['maf_files']['table_origin'],
        'maf_files column_name': config_dict['maf_files']['column_name'],
        'cnv_files column_name': config_dict['cnv_files']['column_name']
    }
    check_required_inputs(required_inputs)

    return config_dict

def collect_data(config_path):
    config_dict = parse_patient_reviewer_input(config_path)

    workspace = config_dict['workspace']
    treatments_fn = config_dict['treatment_file']
    participants_fn = config_dict['clinical_file']['file_name']
    data_path = config_dict['data_path']
    alleliccapseg = config_dict['cnv_files']['column_name']
    unmatched_alleliccapseg = config_dict['cnv_files']['unmatched_column_name']
    maf = config_dict['maf_files']['column_name']
    unmatched_maf = config_dict['maf_files']['unmatched_column_name']
    participant_file = config_dict['saved_files']['participant_file']
    sample_file = config_dict['saved_files']['sample_file']


    sample_cols_values = list(config_dict['sample_columns'].values())
    sample_cols = [col for col in sample_cols_values if col]
    if unmatched_alleliccapseg != '':
        sample_cols.append(unmatched_alleliccapseg)
    if unmatched_maf != '':
        sample_cols.append(unmatched_maf)

    pairs_cols_values = list(config_dict['pairs_columns'].values())
    pairs_cols = [col for col in pairs_cols_values if col]
    pairs_cols.extend([alleliccapseg, maf])

    wm = dalmatian.WorkspaceManager(workspace)

    samples_df = wm.get_samples()
    #samples_df = samples_df[['participant', 'pdb_collection_date_dfd', 'unmatched_alleliccapseg_tsv', 'unmatched_mutation_validator_validated_maf']]
    samples_df = samples_df[sample_cols]

    pairs_df = wm.get_pairs()
    pairs_df = pairs_df[pairs_cols]
    pairs_df.set_index(config_dict['pairs_columns']['sample_id'], inplace=True)

    new_samples_df = samples_df.combine_first(pairs_df)
    if unmatched_alleliccapseg != '':
        new_samples_df.fillna(value={alleliccapseg: new_samples_df[unmatched_alleliccapseg]}, inplace=True)
    if unmatched_maf != '':
        new_samples_df.fillna(value={maf: new_samples_df[unmatched_maf]}, inplace=True)
    new_samples_df.reset_index(inplace=True)
    new_samples_df.rename(columns={
        'index': 'sample_id',
        config_dict['sample_columns']['participant_id']: 'participant_id',
        config_dict['sample_columns']['collection_date']: 'collection_date_dfd',
        alleliccapseg: 'cnv_seg_fn',
        maf: 'maf_fn'
    }, inplace=True)
    new_samples_df.dropna(subset=['cnv_seg_fn', 'maf_fn'], inplace=True)
    new_samples_df.drop(columns=[unmatched_alleliccapseg, unmatched_maf], errors='ignore', inplace=True)

    clinical_df = pd.read_csv(participants_fn, sep='\t')
    # clinical_df file originally has multi index over all the columns
    # make this more robust
    clinical_df.reset_index(inplace=True)
    for col in list(clinical_df):
        clinical_df.rename(columns={col: clinical_df.loc[0, col]}, inplace=True)
    clinical_df.drop(0, inplace=True)
    clinical_df.set_index('participant_id', inplace=True)

    participants_df = wm.get_participants()
    participants_df = participants_df.loc[new_samples_df.participant_id.unique()]
    participants_df.reset_index(inplace=True)
    participants_df = participants_df['participant_id'].to_frame()

    clinical_df_cols = [
        'tumor_molecular_subtype',
        'tumor_morphology',
        'tumor_primary_site',
        'cancer_stage',
        'vital_status',
        'death_date_dfd',
        'follow_up_date',
        'age_at_diagnosis',
        'gender',
        'notes'
    ] or config_dict['clinical_file']['columns']
    clinical_df = clinical_df[[col for col in clinical_df_cols if col in list(clinical_df)]]
    participants_df = participants_df.join(clinical_df, on='participant_id')
    participants_df = participants_df.replace(['unknown', 'not reported'], np.nan)

    treatments_df = pd.read_csv(treatments_fn, sep='\t')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    treatment_files_path = f'{data_path}/preprocess_data/treatments'
    participant_maf_files_path = f'{data_path}/preprocess_data/patient_mafs'

    if not os.path.exists(treatment_files_path):
        os.makedirs(treatment_files_path)
    if not os.path.exists(participant_maf_files_path):
        os.makedirs(participant_maf_files_path)

    for i, participant in enumerate(participants_df.participant_id):

        treatments_file_name = f'{treatment_files_path}/{participant}_treatment.txt'
        if not os.path.exists(treatments_file_name):
            this_p_treatments = treatments_df[treatments_df['participant_id'] == participant]
            if this_p_treatments.shape[0] > 0:
                this_p_treatments.to_csv(treatments_file_name, sep='\t', index=False)

        participants_df.loc[i, 'treatments_fn'] = os.path.normpath(treatments_file_name)

        maf_file_name = f'{participant_maf_files_path}/{participant}_maf.txt'
        if not os.path.exists(maf_file_name):
            this_p_mafs = new_samples_df[new_samples_df['participant_id'] == participant]['maf_fn'].tolist()
            if len(this_p_mafs) > 0:
                this_p_maf = pd.concat([pd.read_csv(maf, sep='\t', encoding = "ISO-8859-1") for maf in this_p_mafs])
                this_p_maf.to_csv(maf_file_name, sep='\t', index=False)

        participants_df.loc[i, 'maf_fn'] = os.path.normpath(maf_file_name)

    participants_df.dropna(subset=['treatments_fn', 'maf_fn'], inplace=True)

    participant_file_name = f'{data_path}/{participant_file}'
    samples_file_name = f'{data_path}/{sample_file}'

    if not os.path.exists(participant_file_name):
        participants_df.to_csv(participant_file_name, sep='\t', index=False)
    if not os.path.exists(samples_file_name):
        new_samples_df.to_csv(samples_file_name, sep='\t', index=False)

    return [new_samples_df, participants_df]

def gen_clinical_data_table(data: PatientSampleData, idx, cols):
    df=data.participant_df
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

    def gen_data(
        self,
        description: str,
        participant_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        annot_df: pd.DataFrame = None,
        annot_col_config_dict: Dict = None,
        history_df: pd.DataFrame = None,
        review_data_annotation_dict: {str: DataAnnotation} = {},
        preprocess_data_dir='.',
        reload_cnv_figs=False,
        index: List = None
    ) -> PatientSampleData:

        """

        Parameters
        ----------
        description: str
            Describe the review session. This is useful if you copy the history of this object to a new review data
            object
        participnat_df
            dataframe containing participant data. this will be the primary dataframe
        sample_df
            dataframe containing sample data
        annot_df: pd.DataFrame
            Dataframe of with previous/prefilled annotations
        annot_col_config_dict: Dict
            Dictionary specifying active annotation columns and validation configurations
        history_df: pd.DataFrame
            Dataframe of with previous/prefilled history
        review_data_annotation_dict
            dictionary containing annotation fields
        preprocess_data_dir
            directory for preproccessing data to be stored
        index: List
            List of values to annotate. participant_df's index

        Returns
        -------
        ReviewData object

        """
        cnv_figs_dir = f'{preprocess_data_dir}/cnv_figs'
        if not os.path.exists(cnv_figs_dir):
            os.makedirs(cnv_figs_dir)
            reload_cnv_figs = True
        else:
            print(f'cnv figs directory already exists: {cnv_figs_dir}')

        if reload_cnv_figs:
            sample_list = sample_df.index.tolist()

            for sample in sample_list:
                output_fn = f'{cnv_figs_dir}/{sample}.cnv_fig.pkl'
                fig, start_trace, end_trace = gen_preloaded_cnv_plot(sample_df, sample)
                pickle.dump(fig, open(output_fn, 'wb'))
                sample_df.loc[sample, 'cnv_plots_pkl'] = output_fn
                sample_df.loc[sample, 'cnv_start_trace'] = start_trace
                sample_df.loc[sample, 'cnv_end_trace'] = end_trace

        rd = PatientSampleData(
            index=participant_df.index,
            description=description,
            participant_df=participant_df,
            sample_df=sample_df,
        )

        return rd

    def set_default_review_data_annotations(self):
        self.add_review_data_annotation('Resistance Explained', DataAnnotation('string',
                                                                                     options=['Mutation', 'CNV',
                                                                                              'Partial/Hypothesized',
                                                                                              'Unknown']))
        self.add_review_data_annotation('Resistance Notes', DataAnnotation('string'))
        self.add_review_data_annotation('Growing Clones',
                                        DataAnnotation('string', validate_input=validate_string_list))
        self.add_review_data_annotation('Shrinking Clones',
                                        DataAnnotation('string', validate_input=validate_string_list))
        self.add_review_data_annotation('Annotations', DataAnnotation('string', options=['Hypermutated',
                                                                                               'Convergent Evolution',
                                                                                               'Strong clonal changes']))
        self.add_review_data_annotation('Selected Tree (idx)', DataAnnotation('int', default=1))
        self.add_review_data_annotation('Other Notes', DataAnnotation('string'))

    def set_default_review_data_annotations_app_display(self):
        self.add_review_data_annotations_app_display('Resistance Explained', 'radioitem')
        self.add_review_data_annotations_app_display('Resistance Notes', 'textarea')
        self.add_review_data_annotations_app_display('Growing Clones', 'text')
        self.add_review_data_annotations_app_display('Shrinking Clones', 'text')
        self.add_review_data_annotations_app_display('Annotations', 'checklist')
        self.add_review_data_annotations_app_display('Selected Tree (idx)', 'number')
        self.add_review_data_annotations_app_display('Other Notes', 'textarea')

    def gen_review_app(
        self,
        preprocess_data_dir,
        custom_colors=[],
        drivers_fn=None,
    ) -> ReviewDataApp:
        """Generate app layout.

        Parameters
        ----------
        custom_colors : list of lists
            specify colummn colors in mutation table with format:
            [[column_id_1, filter_query_1, text_color_1, background_color_1]]
        drivers_fn
            file path to csv file of drivers

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

        if 'build_tree_posterior_fn' and 'cluster_ccfs_fn' in list(self.review_data.data.participant_df):
            app.add_component(gen_phylogic_app_component(), drivers_fn=drivers_fn)

        app.add_component(gen_cnv_plot_app_component(), preprocess_data_dir=preprocess_data_dir)

        return app


    def set_default_autofill(self):
        if 'Phylogic Graphics' in self.app.more_components.items():
            self.add_autofill('Phylogic Graphics', State('tree-dropdown', 'value'), 'Selected Tree (idx)')
