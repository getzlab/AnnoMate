from .ReviewData import ReviewData, DataAnnotation, Data
from .ReviewDataApp import ReviewDataApp, valid_annotation_app_display_types

import pandas as pd
import os
from dash.dependencies import State
from typing import Union, Dict
from abc import ABC, abstractmethod
import pathlib
import pickle


class ReviewerTemplate(ABC):
    
    def __init__(self):
        self.review_data = None
        self.app = None
        self.autofill_dict = {}
        self.annot_app_display_types_dict = {}
    
    @abstractmethod
    def gen_data(self,
                 description: str,
                 annot_df: pd.DataFrame,
                 annot_col_config_dict: Dict,
                 history_df: pd.DataFrame,
                 **kwargs) -> Data:
        """
        Specify type of data object to return and include additional kwargs
        for all the tables required for the Data Object type
        """
        pass

    @abstractmethod
    def set_default_review_data_annotations(self):
        """
        Add annotations to review data object.
        Use self.add_review_data_annotation
        """
        pass

    @abstractmethod
    def gen_review_app(self) -> ReviewDataApp:
        """
        app = ReviewDataApp()
        app.add_component()
        """
        pass

    @abstractmethod
    def set_default_review_data_annotations_app_display(self):
        """
        Define how annotation columns are displayed in the app.
        Use self.add_review_data_annotation()
        """
        pass

    @abstractmethod
    def set_default_autofill(self):
        """
        self.add_autofill()
        """
        pass
    
    # Public methods
    def set_review_data(self,
                        data_pkl_fn: pathlib.Path,
                        description: str,
                        load_existing_data_pkl_fn: Union[str, pathlib.Path] = None,
                        load_existing_exported_data_dir: Union[str, pathlib.Path] = None,
                        annot_df: pd.DataFrame = None,
                        annot_col_config_dict: pd.DataFrame = None,
                        history_df: pd.DataFrame = None,
                        **kwargs):

        if (load_existing_data_pkl_fn is not None) and \
                os.path.exists(load_existing_data_pkl_fn):
            print("Loading data from previous review with pickle file")
            f = open(load_existing_data_pkl_fn, 'rb')
            existing_data = pickle.load(f)
            f.close()

            annot_df = existing_data.annot_df
            annot_col_config_dict = existing_data.annot_col_config_dict
            history_df = existing_data.history_df

        elif (load_existing_exported_data_dir is not None) and \
                os.path.exists(load_existing_exported_data_dir):
            print("Loading data from previous review with exported files")
            annot_df_fn = f'{load_existing_exported_data_dir}/annot_df.tsv'
            history_df_fn = f'{load_existing_exported_data_dir}/history_df.tsv'
            annot_df = pd.read_csv(annot_df_fn, sep='\t')
            history_df = pd.read_csv(history_df_fn, sep='\t')

        self.review_data = ReviewData(data_pkl_fn,
                                      self.gen_data(description,
                                                    annot_df=annot_df,
                                                    annot_col_config_dict=annot_col_config_dict,
                                                    history_df=history_df,
                                                    **kwargs))

    def set_default_review_data_annotations_configuration(self):
        self.set_default_review_data_annotations()
        self.set_default_review_data_annotations_app_display()

    def add_review_data_annotation(self, annot_name: str, review_data_annotation: DataAnnotation):
        self.review_data.add_annotation(annot_name, review_data_annotation)
    
    def set_review_app(self, *args, **kwargs):
        self.app = self.gen_review_app(*args, **kwargs)

    def add_review_data_annotations_app_display(self, name, app_display_type):
        if name not in self.review_data.data.annot_col_config_dict.keys():
            raise ValueError(f"Invalid annotation name '{name}'. "
                             f"Does not exist in review data object annotation table")

        if app_display_type not in valid_annotation_app_display_types:
            raise ValueError(f"Invalid app display type {app_display_type}. "
                             f"Valid options are {valid_annotation_app_display_types}")

        # TODO: check if display type matches annotation type (list vs single value)

        self.annot_app_display_types_dict[name] = app_display_type
        
    def add_autofill(self, component_name: str, fill_value: Union[State, str, float], annot_col: str):
        if component_name not in self.autofill_dict.keys():
            self.autofill_dict[component_name] = {annot_col: fill_value}
        else:
            self.autofill_dict[component_name][annot_col] = fill_value
        
        # verify 
        self.app.gen_autofill_buttons_and_states(self.review_data, self.autofill_dict)

    def run(self, 
            mode='external', 
            host='0.0.0.0', 
            port=8050):
        self.app.run(review_data=self.review_data, 
                     autofill_dict=self.autofill_dict,
                     annot_app_display_types_dict=self.annot_app_display_types_dict,
                     mode=mode,
                     host=host,
                     port=port)
