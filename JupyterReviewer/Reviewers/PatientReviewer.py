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
from dash import Dash, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dalmatian
from typing import Dict, List
import yaml
import re

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
    """Check for required config inputs in list of given inputs"""
    for input in required_inputs:
        if required_inputs[input] == None:
            raise ValueError(f'Required input was not given: {input}')

def parse_patient_reviewer_input(config_path):
    """Parse config file, ensuring all required inputs are given.

    Parameters
    ----------
    config_path: str
        path to config file

    Returns
    -------
    config_dict: dict
        parsed yaml file as a dictionary
    """
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
    """Collect data from config file into sample and participant dataframes.

    Parameters
    ----------
    config_path: str
        path to config file

    Returns
    -------
    combined_samples_df: pd.DataFrame()
        All sample based data from config file
    participants_df: pd.DataFrame()
        All participant based data from config file

    """
    config_dict = parse_patient_reviewer_input(config_path)

    # define config inputs with defaults
    # sample_file = config_dict['saved_files']['sample_file'] if config_dict['saved_files']['sample_file'] else 'samples.txt'
    # participant_file = config_dict['saved_files']['participant_file'] if config_dict['saved_files']['participant_file'] else 'participants.txt'

    default_clinical_cols = [
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
    ]
    input_clinical_cols = [col for col in list(config_dict['clinical_file']['columns'].values()) if col]
    if input_clinical_cols:
        clinical_cols = input_clinical_cols
    else:
        clinical_cols = default_clinical_cols
    if config_dict['clinical_file']['additional_columns']:
        clinical_cols.extend(config_dict['clinical_file']['additional_columns'])

    default_sample_cols = {
        'collection_date': 'pdb_collection_date_dfd',
        'cram_bam_columns': [
            'gpdw_DNA_WES_agilent_cram_or_bam_path',
            'gpdw_DNA_WES_icev1_cram_or_bam_path',
            'gpdw_DNA_WES_twistv1_cram_or_bam_path',
            'gpdw_DNA_WGS_cram_or_bam_path'
        ],
        'participant_id': 'participant',
        'ploidy': None,
        'preservation_method': 'pdb_preservation_method',
        'purity': None
    }
    input_sample_cols = config_dict['sample_columns']['columns']
    sample_cols=[]
    for col in input_sample_cols:
        if input_sample_cols[col]:
            sample_cols.extend(input_sample_cols[col]) if isinstance(input_sample_cols[col], list) else sample_cols.append(input_sample_cols[col])
        elif default_sample_cols[col]:
            sample_cols.extend(default_sample_cols[col]) if isinstance(default_sample_cols[col], list) else sample_cols.append(default_sample_cols[col])
    if config_dict['sample_columns']['additional_columns']:
        sample_cols.extend(config_dict['sample_columns']['additional_columns'])

    default_pairs_cols = {
        'sample_id': 'case_sample',
        'participant_id': 'participant',
        'purity': None,
        'ploidy': None
    }
    input_pairs_cols = config_dict['pairs_columns']
    pairs_cols = []
    for col in input_pairs_cols:
        if input_pairs_cols[col]:
            pairs_cols.append(input_pairs_cols[col])
        elif default_pairs_cols[col]:
            pairs_cols.append(default_pairs_cols[col])

    # define required config inputs
    workspace = config_dict['workspace']
    data_path = config_dict['data_path']
    clinical_fn = config_dict['clinical_file']['file_name']

    participant_cols = []
    maf = config_dict['maf_files']['column_name']
    maf_table_origin = config_dict['maf_files']['table_origin']
    if maf_table_origin == 'sample':
        pairs_cols.append(maf)
    elif maf_table_origin == 'participant':
        participant_cols.append(maf)
    else:
        raise ValueError(f'maf_table_origin must be sample or participant, not {maf_table_origin}')

    alleliccapseg = config_dict['cnv_files']['column_name']
    pairs_cols.append(alleliccapseg)

    # define config inputs that are empty if not specified
    treatments_fn = config_dict['treatment_file']

    unmatched_alleliccapseg = config_dict['cnv_files']['unmatched_column_name']
    if unmatched_alleliccapseg:
        sample_cols.append(unmatched_alleliccapseg)

    unmatched_maf = config_dict['maf_files']['unmatched_column_name']
    if unmatched_maf:
        sample_cols.append(unmatched_maf)

    cluster_ccfs = config_dict['phylogic_files']['clusters_file']
    if cluster_ccfs:
        participant_cols.append(cluster_ccfs)
    trees = config_dict['phylogic_files']['trees_file']
    if trees:
        participant_cols.append(trees)

    # begin pulling data from terra based off of above specifications
    wm = dalmatian.WorkspaceManager(workspace)

    samples_df = wm.get_samples()
    samples_df = samples_df[sample_cols]
    samples_df.replace('', np.NaN, inplace=True)

    pairs_df = wm.get_pairs()
    pairs_df = pairs_df[pairs_cols]
    pairs_df.set_index(config_dict['pairs_columns']['sample_id'], inplace=True)

    # combine samples and pairs on sample_id, still appending samples not in pairs
    # index is sample_id
    combined_samples_df = samples_df.combine_first(pairs_df)
    # merge all alleliccapseg and maf files between samples and pairs into one column
    if unmatched_alleliccapseg:
        combined_samples_df.fillna(value={alleliccapseg: combined_samples_df[unmatched_alleliccapseg]}, inplace=True)
    if unmatched_maf:
        combined_samples_df.fillna(value={maf: combined_samples_df[unmatched_maf]}, inplace=True)
    combined_samples_df.reset_index(inplace=True)
    # rename all input columns to eliminate ambiguity throughout the code
    combined_samples_df.rename(columns={
        'index': 'sample_id',
        config_dict['sample_columns']['columns']['participant_id']: 'participant_id',
        config_dict['sample_columns']['columns']['collection_date']: 'collection_date_dfd',
        config_dict['sample_columns']['columns']['preservation_method']: 'preservation_method',
        config_dict['pairs_columns']['purity']: 'wxs_purity',
        config_dict['pairs_columns']['ploidy']: 'wxs_ploidy',
        alleliccapseg: 'cnv_seg_fn',
        maf: 'maf_fn'
    }, inplace=True)
    combined_samples_df.dropna(subset=['cnv_seg_fn'], inplace=True)
    if 'maf_fn' in list(combined_samples_df):
        combined_samples_df.dropna(subset=['maf_fn'], inplace=True)
    # now that all alleliccapseg and maf data is merged into one column, these two can be dropped
    # errors=ignore ignored error if unmatched alleliccapseg and maf are not present
    combined_samples_df.drop(columns=[unmatched_alleliccapseg, unmatched_maf], errors='ignore', inplace=True)

    # force purity and ploidy columns to be type float
    if 'wxs_purity' and 'wxs_ploidy' in combined_samples_df:
        combined_samples_df['wxs_purity'] = combined_samples_df['wxs_purity'].astype(float)
        combined_samples_df['wxs_ploidy'] = combined_samples_df['wxs_ploidy'].astype(float)

    clinical_df = pd.read_csv(clinical_fn, sep='\t', comment='#')
    clinical_df.set_index('participant_id', inplace=True)

    participants_df = wm.get_participants()
    # limit participants_df to only participants that have samples
    participants_df = participants_df.loc[combined_samples_df.participant_id.unique()]
    participants_df.reset_index(inplace=True)

    if participant_cols:
        participant_cols.append('participant_id')
        participants_df = participants_df[participant_cols]
        participants_df.rename(columns={
            maf: 'maf_fn',
            cluster_ccfs: 'cluster_ccfs_fn',
            trees: 'build_tree_posterior_fn'
        }, inplace=True)
    else:
        participants_df = participants_df['participant_id'].to_frame()

    clinical_df = clinical_df[clinical_cols]
    participants_df = participants_df.join(clinical_df, on='participant_id')

    treatments_df = pd.read_csv(treatments_fn, sep='\t', comment='#')

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

        if maf_table_origin == 'sample':
            maf_file_name = f'{participant_maf_files_path}/{participant}_maf.txt'
            if not os.path.exists(maf_file_name):
                this_p_mafs = combined_samples_df[combined_samples_df['participant_id'] == participant]['maf_fn'].tolist()
                if len(this_p_mafs) > 0:
                    this_p_maf = pd.concat([pd.read_csv(maf, sep='\t', encoding = "ISO-8859-1") for maf in this_p_mafs])
                    this_p_maf.to_csv(maf_file_name, sep='\t', index=False)

            participants_df.loc[i, 'maf_fn'] = os.path.normpath(maf_file_name)

    participants_df.dropna(subset=['treatments_fn', 'maf_fn'], inplace=True)

    # participant_file_name = f'{data_path}/{participant_file}'
    # samples_file_name = f'{data_path}/{sample_file}'

    # if not os.path.exists(participant_file_name):
    #     participants_df.to_csv(participant_file_name, sep='\t', index=False)
    # if not os.path.exists(samples_file_name):
    #     combined_samples_df.to_csv(samples_file_name, sep='\t', index=False)

    return [combined_samples_df, participants_df]

def gen_clinical_data_table(df, idx):
    """Format clinical data into a dash DataTable.

    Parameters
    ----------
    df
        Participant level DataFrame
    idx

    Returns
    -------
    dash_table.DataTable()

    """

    default_cols = {
        'tumor_molecular_subtype': 'Molecular Subtype',
        'tumor_morphology': 'Morphology',
        'tumor_primary_site': 'Primary Site',
        'cancer_stage': 'Stage',
        'vital_status': 'Vital Status',
        'death_date_dfd': 'Death Date (dfd)',
        'follow_up_date': 'Last Follow Up Date (dfd)',
        'age_at_diagnosis': 'Age (at diagnosis)',
        'gender': 'Gender',
        'notes': 'Notes'
    }

    participant_cols = list(df)
    clinical_cols = [col for col in participant_cols if not re.search('fn$|id$|pickle$', col)]

    this_participant_df = df.loc[idx, clinical_cols].to_frame()
    this_participant_df.reset_index(inplace=True)
    this_participant_df['index'] = this_participant_df['index'].apply(lambda x: default_cols[x] if x in default_cols else x.replace('_', ' '))
    this_participant_df.set_index('index')
    this_participant_df = this_participant_df.replace(['unknown', 'not reported'], np.NaN)
    this_participant_df.dropna(inplace=True)

    return [dash_table.DataTable(
        data=this_participant_df.to_dict('records'),
        columns=[
            {'name': '', 'id': 'index'},
            {'name': idx, 'id': idx}]
    )]

def gen_sample_data_table(df, idx):
    """Format sample data into a dash DataTable.

    Parameters
    ----------
    df
        Sample level DataFrame
    idx

    Returns
    -------
    dash_table.DataTable()

    """
    default_cols = {
        'sample_id': 'Sample ID',
        'collection_date_dfd': 'Collection Date (dfd)',
        'sample_type': 'Sample Type',
        'preservation_method': 'Preservation Method',
        'bait_set': 'Bait Set',
        'wxs_purity': 'Purity',
        'wxs_ploidy': 'Ploidy'
    }

    cram_bam_columns = [col for col in list(df) if re.search('cram_or_bam', col)]
    df['sample_type'] = np.NaN
    df['bait_set'] = np.NaN
    for sample in df.index:
        for col in cram_bam_columns:
            if not pd.isnull(df.loc[sample, col]):
                if not pd.isnull(df.loc[sample, 'sample_type']) and not re.search(df.loc[sample, 'sample_type'], col):
                    df.loc[sample, 'sample_type'] = 'WES, WGS'
                else:
                    df.loc[sample, 'sample_type'] = 'WES' if re.search('WES', col) else 'WGS'

                if not pd.isnull(df.loc[sample, 'bait_set']) and not re.search('WGS', col):
                    df.loc[sample, 'bait_set'] += ', TWIST' if re.search('twist', col) else ', ICE' if re.search('ice', col) else ', Agilent'
                elif pd.isnull(df.loc[sample, 'bait_set']):
                    df.loc[sample, 'bait_set'] = 'TWIST' if re.search('twist', col) else 'ICE' if re.search('ice', col) else 'Agilent' if re.search('agilent', col) else np.nan

    df.reset_index(inplace=True)
    df.set_index('participant_id', inplace=True)
    sample_cols = [col for col in list(df) if not (re.search('fn$|cram_or_bam|pickle$', col))]
    this_sample_df = df.loc[idx, sample_cols]
    this_sample_df.reset_index(drop=True, inplace=True)
    this_sample_df.dropna(axis=1, how='all', inplace=True)

    formatted_cols = [{'name': default_cols[col], 'id': col} if col in default_cols else {'name': col.replace('_', ' '), 'id': col} for col in sample_cols ]

    df.reset_index(inplace=True)
    df.set_index('sample_id', inplace=True)

    return [dash_table.DataTable(
        data=this_sample_df.to_dict('records'),
        columns=formatted_cols
    )]


def gen_clinical_sample_data_table(data: PatientSampleData, idx):
    """Clinical and sample data callback function"""
    df = data.participant_df
    samples_df = data.sample_df

    return [
        gen_clinical_data_table(df, idx),
        gen_sample_data_table(samples_df, idx)
    ]


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
        load_figs_mafs=True,
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
        load_figs_mafs: bool
            Should cnv figures and maf dataframes be computed again? (default True)
        index: List
            List of values to annotate. participant_df's index

        Returns
        -------
        ReviewData object

        """
        cnv_figs_dir = f'{preprocess_data_dir}/cnv_figs'  # todo make this logic better - what if all files don't exist?
        if not os.path.exists(cnv_figs_dir):
            os.makedirs(cnv_figs_dir)
            load_figs_mafs = True
        else:
            print(f'cnv figs directory already exists: {cnv_figs_dir}')

        maf_dir = f'{preprocess_data_dir}/maf_df'
        if not os.path.exists(maf_dir):
            os.makedirs(maf_dir)
            load_figs_mafs = True
        else:
            print(f'Maf directory already exists: {maf_dir}')

        if load_figs_mafs:
            participant_list = participant_df.index.tolist()

            sample_cnv_list = []
            participant_maf_list = []
            for participant_id in participant_list:
                sample_cnv_series, participant_maf_series = gen_preloaded_cnv_plot(participant_df, participant_id,
                                                                                   sample_df, preprocess_data_dir)
                sample_cnv_list.append(sample_cnv_series)
                participant_maf_list.append(participant_maf_series)

            sample_cnv_list = pd.concat(sample_cnv_list)
            participant_maf_list = pd.concat(participant_maf_list)

            participant_df['maf_df_pickle'] = participant_maf_list
            sample_df['cnv_fig_pickle'] = sample_cnv_list
        else:
            pass  # todo get pickle locations

        rd = PatientSampleData(
            index=participant_df.index.tolist(),
            description=description,
            participant_df=participant_df,
            sample_df=sample_df,
        )

        return rd

    def set_default_review_data_annotations(self):
        """Set default annotation sections in the app"""
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
        """Set the display of the components generated in set_default_review_data_annotations"""
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
        custom_colors=None,
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
            'Clinical + Sample Data',
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            id='clinical-data-component',
                            children=[dash_table.DataTable(columns=[])]
                        )
                    ], width=6),
                    dbc.Col([
                        html.Div(
                            id='sample-data-component',
                            children=[dash_table.DataTable(columns=[])]
                        )
                    ], width=6)
                ])
            ]),
            callback_output=[
                Output('clinical-data-component', 'children'),
                Output('sample-data-component', 'children')
            ],
            new_data_callback=gen_clinical_sample_data_table
        ))

        app.add_component(gen_mutation_table_app_component(), custom_colors=custom_colors)

        if 'build_tree_posterior_fn' and 'cluster_ccfs_fn' in list(self.review_data.data.participant_df):
            app.add_component(gen_phylogic_app_component(), drivers_fn=drivers_fn)

        app.add_component(gen_cnv_plot_app_component(), preprocess_data_dir=preprocess_data_dir)

        return app


    def set_default_autofill(self):
        """Set default autofill functionality for annotations """
        if 'build_tree_posterior_fn' and 'cluster_ccfs_f' in list(self.review_data.data.participant_df):
            self.add_autofill('Phylogic Tree', State('tree-dropdown', 'value'), 'Selected Tree (idx)')
