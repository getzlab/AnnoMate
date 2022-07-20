import pandas as pd
from datetime import datetime
import os
import numpy as np
import warnings
import pickle
from pathlib import Path
from typing import List, Dict, Union

valid_annotation_types = ["multi", "float", "int", "string"]


class DataAnnotation:
    
    def __init__(self,
                 annot_value_type: str,
                 options: List = None,
                 validate_input=None,
                 default=None):
        """
        Configure annotation type, validation, and options

        Parameters
        ----------
        annot_value_type: str
            value for the annotation type. Determines dtype of the dataframe column
        options: List
            list of values inputs are allowed to be. For CHECKLIST, RADIOITEM, and DROPDOWN
        validate_input: func
            a custom function to verify annotation input. Takes a single input and returns a boolean
        """

        if annot_value_type not in valid_annotation_types:
            raise ValueError(
                f'annot_type {annot_value_type} is not a valid annotation value type. '
                f'Valid types are {valid_annotation_types}')

        self.annot_value_type = annot_value_type
        self.options = options
        self.validate_input = validate_input
        self.default = default
        
    def validate(self, x):
        if self.options is not None:
            for item in np.array(x).flatten():
                if item not in self.options and item != '':
                    raise ValueError(f'Input {item} is not in the specified options {self.options}')
                
        if self.validate_input is not None:
            if not self.validate_input(x):
                raise ValueError(f'Input {x} is invalid')
        
    def __str__(self):
        return str(self.__dict__)


class Data:
    def __init__(self,
                 index: List,
                 description: str,
                 annot_df: pd.DataFrame = None,
                 annot_col_config_dict: Dict = None,
                 history_df: pd.DataFrame = None):
        """
        Data object to store data to review and tables to store annotation data.

        Parameters
        ----------
        index: List
            List of values to annotate. Index of the annotation table

        description: str
            describe the review session. This is useful if you copy the history of this object to a new review data
            object

        annot_df: pd.DataFrame
            Dataframe of with previous/prefilled annotations

        annot_col_config_dict: Dict
            Dictionary specifying active annotation columns and validation configurations

        history_df: pd.DataFrame
            Dataframe of with previous/prefilled history
        """

        if len(index) != len(set(index)):
            raise ValueError("Indices are not unique")

        self.index = index
        self.description = description

        self.annot_col_config_dict = annot_col_config_dict if annot_col_config_dict is not None else dict()
        annot_cols = list(self.annot_col_config_dict.keys())
        if annot_df is not None:
            annot_cols = list(set(annot_df.columns.tolist() + annot_cols))
            self.annot_df = pd.DataFrame(index=index, columns=annot_cols)
            fill_annot_index = annot_df.index[annot_df.index.isin(index)]
            for col in annot_df.columns:
                self.annot_df.loc[fill_annot_index, col] = annot_df.loc[fill_annot_index, col]
        else:
            self.annot_df = pd.DataFrame(index=index, columns=annot_cols)

        self.history_df = history_df.loc[history_df['index'].isin(index)] if history_df is not None else pd.DataFrame(
            columns=['index', 'timestamp', 'source_data_fn'])


class ReviewData:
    
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

        new_annot_data = {annot_name: ann for annot_name, ann in annot_col_config_dict.items() if
                          annot_name not in self.data.annot_col_config_dict.keys()}
        
        for name, ann in new_annot_data.items():
            if not isinstance(ann, DataAnnotation):
                raise ValueError(f'Annotation name {ann} has invalid value {ann}. '
                                 f'Value in dictionary must be a DataAnnotation object')
            self.data.annot_col_config_dict[name] = ann

        self.data.annot_df[list(new_annot_data.keys())] = np.nan
        self.data.history_df[list(new_annot_data.keys())] = np.nan
        
        for name, annot_data in new_annot_data.items():
            if annot_data.annot_value_type == 'multi':
                self.data.annot_df[name] = self.data.annot_df[name].astype(object)
            elif annot_data.annot_value_type == 'float':
                self.data.annot_df[name] = self.data.annot_df[name].astype(float)
            elif annot_data.annot_value_type == 'string':
                self.data.annot_df[name] = self.data.annot_df[name].astype(str)

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
