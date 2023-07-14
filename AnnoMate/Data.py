from typing import List, Dict
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Data(ABC):

    @abstractmethod
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
        if annot_value_type == 'multi' and options is not None:
            for x in options:
                if ',' in x or "'" in x:
                    raise ValueError("List options cannot contain an apostrophe or comma, due to issues parsing the string.\n"
                                     f"Please remove the offending character (' or ,) from option {x}.")

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


def validate_annot_data(data_annot: DataAnnotation, x):
        
    if (x != '') and (not pd.isna([x]).all()) and (x is not None):

        if data_annot.options is not None:
            x = x if data_annot.annot_value_type == 'multi' else [x]
            for item in x:
                if (item not in data_annot.options): # and (item != '') and (not pd.isna(item)) and (item is not None):
                    raise ValueError(f'Input "{item}" is not in the specified options {data_annot.options}')

        if data_annot.validate_input is not None:
            if not data_annot.validate_input(x):
                raise ValueError(f'Input "{x}" is invalid')
