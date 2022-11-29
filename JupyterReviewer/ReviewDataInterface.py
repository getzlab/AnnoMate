import pandas as pd
from datetime import datetime
import os
import numpy as np
import warnings
import pickle
from pathlib import Path
from typing import List, Dict, Union
from JupyterReviewer.Data import Data, DataAnnotation


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
        
        new_annot_names = [annot_name for annot_name, ann in annot_col_config_dict.items() if annot_name not in self.data.annot_col_config_dict.keys()]
        existing_annot_names = [annot_name for annot_name, ann in annot_col_config_dict.items() if annot_name in self.data.annot_col_config_dict.keys()]
        
        new_annot_data = {annot_name: annot_col_config_dict[annot_name] for annot_name in new_annot_names}
        for annot_name, ann in new_annot_data.items():
            if not isinstance(ann, DataAnnotation):
                raise ValueError(f'Annotation name {annot_name} has invalid value {ann} of type {type(ann)}. '
                                 f'Value in dictionary must be a DataAnnotation object')
            self.data.annot_col_config_dict[annot_name] = ann
            
        for existing_annot_name in existing_annot_names:
            try:
                for idx, r in self.data.annot_df.iterrows():
                    self.validate_annot_data(annot_col_config_dict[existing_annot_name], r[existing_annot_name])
            except ValueError as e:
                raise ValueError(
                    f'Existing data in annotation table (annot_df) column "{existing_annot_name}" are not compatible with new DataAnnotation configuration. '
                    f'Original message: {e}'
                )
                
            # Replace    
            self.data.annot_col_config_dict[existing_annot_name] = annot_col_config_dict[existing_annot_name]
            

        self.data.annot_df[list(new_annot_data.keys())] = np.nan
        self.data.history_df[list(new_annot_data.keys())] = np.nan
        
        for name, annot_data in new_annot_data.items():
            if annot_data.annot_value_type == 'multi':
                self.data.annot_df[name] = self.data.annot_df[name].fillna('').astype(object)
            elif annot_data.annot_value_type == 'float':
                self.data.annot_df[name] = self.data.annot_df[name].astype(float)
            elif annot_data.annot_value_type == 'string':
                self.data.annot_df[name] = self.data.annot_df[name].fillna('').astype(str)

        self.save_data()
        
        
    def validate_annot_data(self, data_annot: DataAnnotation, x):
        
        if data_annot.options is not None:
            for item in np.array(x).flatten():
                if (item not in data_annot.options) and (item != '') and (not pd.isna(item)) and (item is not None):
                    raise ValueError(f'Input "{item}" is not in the specified options {data_annot.options}')

        if data_annot.validate_input is not None:
            if not data_annot.validate_input(x):
                raise ValueError(f'Input "{x}" is invalid')
        
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
            self.data.annot_df.loc[data_idx, list(dictionary.keys())] = list(dictionary.values())
            dictionary['timestamp'] = datetime.today()
            dictionary['index'] = data_idx
            dictionary['source_data_fn'] = self.data_pkl_fn
            self.data.history_df = pd.concat([self.data.history_df, pd.Series(dictionary).to_frame().T])
            self.save_data()
        else:
            pass
            
    def export_data(self, path: Union[str, Path]):
        """
        Export tables in self.data to tsv files in specified directory

        Parameters
        ----------
        path: Union[str, Path]
            local or gsurl path to directory to save object's dataframe objects
        """

        for attribute_name in self.data.__dict__:
            x = getattr(self.data, attribute_name)
            if isinstance(x, pd.DataFrame):
                x.to_csv(f'{path}/{attribute_name}.tsv', sep='\t')
