import pandas as pd
from datetime import datetime
import os
import numpy as np
import warnings
import pickle
from pathlib import Path
from typing import List, Dict, Union
from AnnoMate.Data import Data, DataAnnotation, validate_annot_data


class ReviewDataInterface:
    
    def __init__(self,
                 data_pkl_fn: Union[str, Path],
                 data: Data):
        """
        Object that saves, loads, and edits Data objects

        Parameters
        ----------
        data_pkl_fn: Union[str, Path]
            pickle file to save/load data object from
        data: Data
            data object with the data to review

        Notes
        -----

        If data_pkl_fn already exists, it will only load that file and ignore whatever the parameter data is.
        This is to prevent accidentally overwriting annotations and the data being currently reviewed.
        """
        self.data_pkl_fn = data_pkl_fn
        if os.path.exists(data_pkl_fn):
            f = open(data_pkl_fn, 'rb')
            self.data = pickle.load(f)
            f.close()
            warnings.warn(f"Loading existing data pkl file")
        else:
            self.data = data

        self.save_data()

    def save_data(self):
        """
        Saves Data object to pickle file
        """
        f = open(self.data_pkl_fn, 'wb')
        pickle.dump(self.data, f, 2)
        f.close()
        
    def add_annotation(self,
                       annot_name: str,
                       data_annot: DataAnnotation):
        """
        Adds annotation data to the Data object

        Parameters
        ----------
        annot_name: str
            Name of the annotation that will be a column in the Data object's annot_df dataframe
        data_annot: DataAnnotation
            a DataAnnotation to add to the review data object
        """
        self._add_annotations({annot_name: data_annot})
    
    def _add_annotations(self, annot_col_config_dict: Dict):
        """
        Adds or updates the configuration for annotations to be collected in the Data object

        Parameters
        ----------
        annot_col_config_dict: Dict
            A dictionary where the key is the name of the annotation (column) and the value is a DataAnnotation object
        """

        existing_annot_names = [
            annot_name for annot_name, ann in annot_col_config_dict.items() if annot_name in self.data.annot_df.columns
        ]
        new_annot_names = [
            annot_name for annot_name, ann in annot_col_config_dict.items() if annot_name not in existing_annot_names
        ]

        
        # Check new annotations
        new_data_annot = {annot_name: annot_col_config_dict[annot_name] for annot_name in new_annot_names}
        for annot_name, data_annot in new_data_annot.items():
            if not isinstance(data_annot, DataAnnotation):
                raise ValueError(
                    f'Annotation name {annot_name} has invalid value {data_annot} of type {type(data_annot)}. '
                    f'Value in dictionary must be a DataAnnotation object'
                )
            self.data.annot_col_config_dict[annot_name] = data_annot
            
        # update existing annotations
        for existing_annot_name in existing_annot_names:
            try:
                    
                for idx, r in self.data.annot_df.iterrows():
                    validate_annot_data(annot_col_config_dict[existing_annot_name], r[existing_annot_name])
                    
            except ValueError as e:
                raise ValueError(
                    f'Existing data in annotation table (annot_df) column "{existing_annot_name}" are not compatible with new DataAnnotation configuration. '
                    f'Original message: {e}'
                )
                
            # Replace    
            self.data.annot_col_config_dict[existing_annot_name] = annot_col_config_dict[existing_annot_name]
            
            
        self.data.annot_df[list(new_data_annot.keys())] = np.nan
        self.data.history_df[list(new_data_annot.keys())] = np.nan
        
        # Set types
        for name, data_annot in annot_col_config_dict.items():
            try:
                if data_annot.annot_value_type == 'multi':
                    self.data.annot_df[name] = self.data.annot_df[name].fillna('').astype(object)
                elif data_annot.annot_value_type == 'float':
                    self.data.annot_df[name] = self.data.annot_df[name].astype(float)
                elif data_annot.annot_value_type == 'string':
                    self.data.annot_df[name] = self.data.annot_df[name].fillna('').astype(str)
            except ValueError as e:
                raise ValueError(
                    f'Annotation "{name}" has values that are not compatible with new annot_value_type {data_annot.annot_value_type}. '
                    f'If you mean to change the datatype, consider (1) resetting the reviewer if no annotations were made already, '
                    f'or (2) create a new annotation. '
                    f'Full error: {e}'
                )

        self.save_data()
        
    def _update(self, data_idx, dictionary: Dict):
        """
        Update data annotation table with values in dictionary at index data_idx

        Parameters
        ----------
        data_idx:
            Index in self.data.annot_df
        dictionary: Dict
            A dictionary with keys that exist in self.data.annot_df.columns, and values to put in self.data.annot_df
            at data_idx
        """
        if list(self.data.annot_df.loc[data_idx, list(dictionary.keys())].values) != list(dictionary.values()):
            
            with warnings.catch_warnings():
                
                # Catching warning where the annotation value is "multi" (a list type)
                warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
                
                self.data.annot_df.loc[data_idx, list(dictionary.keys())] = list(dictionary.values())
                dictionary['timestamp'] = datetime.today()
                dictionary['index'] = data_idx
                dictionary['source_data_fn'] = self.data_pkl_fn
                self.data.history_df = pd.concat([self.data.history_df, pd.Series(dictionary).to_frame().T])
                self.save_data()
        else:
            pass
            
    def export_data(self, path: Union[str, Path], attributes_to_export: List = None, verbose=True):
        """
        Export tables in self.data to tsv files in specified directory

        Parameters
        ----------
        path: Union[str, Path]
            local or gsurl path to directory to save object's dataframe objects
        attributes_to_export: List
            Specify which attributes to export
        """
        attributes_to_export = self.data.__dict__.keys() if attributes_to_export is None else attributes_to_export 

        for attribute_name in attributes_to_export:
            x = getattr(self.data, attribute_name)
            if isinstance(x, pd.DataFrame):
                fn = f'{path}/{attribute_name}.tsv'
                if verbose: print(f'Saving {attribute_name} to {fn}')
                x.to_csv(fn, sep='\t')
            else:
                if verbose: print(f'{attribute_name} is not a dataframe. Not exporting.')
