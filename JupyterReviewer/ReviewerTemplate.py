from .ReviewDataInterface import ReviewDataInterface, DataAnnotation, Data
from .ReviewDataApp import ReviewDataApp, valid_annotation_app_display_types

import pandas as pd
import os
from dash.dependencies import State
from typing import Union, Dict, List
from abc import ABC, abstractmethod
import pathlib
import pickle
import inspect
import traceback


def make_docstring(object_type_name, func1_doc, func2):
    func2_doc = f"{object_type_name}.{func2.__name__}" + str(inspect.signature(func2)) + func2.__doc__
    if func2_doc not in func1_doc:
        return func1_doc + func2_doc
    else:
        return func1_doc


class ReviewerTemplate(ABC):
    
    def __init__(self):
        self.review_data_interface = None
        self.app = None
        self.autofill_dict = {}
        self.annot_app_display_types_dict = {}
        type(self).add_review_data_annotation.__doc__ = \
            make_docstring(object_type_name="ReviewDataInterface",
                           func1_doc=type(self).add_review_data_annotation.__doc__,
                           func2=ReviewDataInterface.add_annotation)

        try:
            type(self).set_review_app.__doc__ = \
                make_docstring(object_type_name=type(self).__name__,
                               func1_doc=type(self).set_review_app.__doc__,
                               func2=type(self).gen_review_app)

            type(self).set_review_data.__doc__ = \
                make_docstring(object_type_name=type(self).__name__,
                               func1_doc=type(self).set_review_data.__doc__,
                               func2=type(self).gen_data)
        except TypeError as e:
            raise TypeError(
                f"Docstrings may be missing from gen_review_app and/or gen_data.\n"
                f"Full trace:\n {traceback.format_exc()}")


    @abstractmethod
    def gen_data(self,
                 description: str,
                 annot_df: pd.DataFrame = None,
                 annot_col_config_dict: Dict = None,
                 history_df: pd.DataFrame = None,
                 index: List = None,
                 *args,
                 **kwargs) -> Data:
        """
        Specify type of data object to return and include additional kwargs
        for all the tables required for the Data Object type

        Returns
        -------
        Data
            Object that stores data to be reviewed and corresponding annotation and history

        """
        pass

    @abstractmethod
    def set_default_review_data_annotations(self):
        """Add annotations to review data object.
        Notes
        -----
        Call self.add_review_data_annotation() to set custom annotation configurations

        """
        pass

    @abstractmethod
    def gen_review_app(self, *args, **kwargs) -> ReviewDataApp:
        """
        Generates a ReviewDataApp object

        Returns
        -------
        ReviewDataApp
            App object that will display the dashboard

        Notes
        -----
        1. Instatiate a ReviewDataApp
        2. Add components

            ```
            app = ReviewDataApp()
            app.add_component(AppComponent(...))
            ```
        """
        pass

    @abstractmethod
    def set_default_review_data_annotations_app_display(self):
        """Define how annotation columns are displayed in the app.
        Notes
        -----
        Call self.add_review_data_annotations_app_display() to set custom display options

        """
        pass

    @abstractmethod
    def set_default_autofill(self):
        """Set which component values can be used for auto-filling input form

        Notes
        -----
        Call self.add_autofill() to set custom autofill options

        """
        pass
    
    # Public methods
    def set_review_data(self,
                        data_pkl_fn: pathlib.Path,
                        description: str = None,
                        annot_df: pd.DataFrame = None,
                        annot_col_config_dict: pd.DataFrame = None,
                        history_df: pd.DataFrame = None,
                        load_existing_data_pkl_fn: Union[str, pathlib.Path] = None,
                        load_existing_exported_data_dir: Union[str, pathlib.Path] = None,
                        **kwargs):
        """Sets the review session ReviewData Object.

        Parameters
        ----------
        data_pkl_fn : Union[str, Path]
            path to pickle file to save data for current review session

        description : str,
            description of the data being reviewed

        annot_df : pd.DataFrame, optional
            dataframe of annotations from previous review sessions.

        annot_col_config_dict : dict(), optional
            dictionary defining the DataAnnotation configurations for the columns of annot_df

        history_df : pd.DataFrame, optional
            dataframe of history from previous review sessions.

        load_existing_data_pkl_fn : Union[str, Path], optional
            path to pickle file of a previous review session's data object.

        load_existing_exported_data_dir : Union[str, Path]
            path to a directory with exported annotation and history
            tables from a previous review session's data object

        **kwargs: dict
                  See additional parameters from self.gen_data() below

        """

        if os.path.exists(data_pkl_fn):
            f = open(data_pkl_fn, 'rb')
            data = pickle.load(f)
        else:
            if description is None:
                raise ValueError(f'description is None. Provide a description if you are setting a new data object.')
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

            data = self.gen_data(
                description,
                annot_df=annot_df,
                annot_col_config_dict=annot_col_config_dict,
                history_df=history_df,
                **kwargs
            )

        self.review_data_interface = ReviewDataInterface(
            data_pkl_fn=data_pkl_fn,
            data=data,
        )

    def set_default_review_data_annotations_configuration(self):
        """
        Sets preconfigured annotation columns and display settings

        Notes
        -----
        Keep default annotation columns but change display options

            ```
            my_reviewer.set_default_review_data_annotations()

            # custom settings for each
            my_reviewer.add_review_data_annotations_app_display(...)
            my_reviewer.add_review_data_annotations_app_display(...)
            ...
            ```

            or overwrite the defaults

            ```
            my_reviewer.set_default_review_data_annotations_configuration()

            # custom settings for each
            my_reviewer.add_review_data_annotations_app_display(...)
            my_reviewer.add_review_data_annotations_app_display(...)
            ...
            ```

        Add or update annotation configurations
            ```
            my_reviewer.set_default_review_data_annotations()
            my_reviewer.add_review_data_annotations(...)
            my_reviewer.add_review_data_annotations(...)
            ```
            or, completely customize
            ```
            # skip my_reviewer.set_default_review_data_annotations()
            my_reviewer.add_review_data_annotations(...)
            my_reviewer.add_review_data_annotations(...)
            ```

        """
        self.set_default_review_data_annotations()
        self.set_default_review_data_annotations_app_display()

    def add_review_data_annotation(self, annot_name: str, review_data_annotation: DataAnnotation):
        """
        See ReviewData.add_annotation

        """
        self.review_data_interface.add_annotation(annot_name, review_data_annotation)
    
    def set_review_app(self, *args, **kwargs):
        """
        Sets the reviewer's ReviewDataApp.

        Parameters
        ----------
        *args:
            See self.gen_review_app
        **kwargs:
            See self.gen_review_app

        """
        self.app = self.gen_review_app(*args, **kwargs)

    def add_review_data_annotations_app_display(self,
                                                annot_name: str,
                                                app_display_type: str):
        """
        Set display type for annotations

        Parameters
        ----------
        annot_name: str
            Name of an annotation column in self.review_data.data annotation table.
            The annotation must be configured in self.review_data.data.annot_col_config_dict
        app_display_type: str
            Type of input display for the annotation. See ReviewDataApp.valid_annotation_app_display_types

        """
        if annot_name not in self.review_data_interface.data.annot_col_config_dict.keys():
            raise ValueError(f"Invalid annotation name '{annot_name}'. "
                             f"Does not exist in review data object annotation table")

        if app_display_type not in valid_annotation_app_display_types:
            raise ValueError(f"Invalid app display type {app_display_type}. "
                             f"Valid options are {valid_annotation_app_display_types}")

        # TODO: check if display type matches annotation type (list vs single value)

        self.annot_app_display_types_dict[annot_name] = app_display_type
        
    def add_autofill(self, autofill_button_name: str, fill_value: Union[State, str, float], annot_col: str):
        """
        Configures the option to use the state of a component in the ReviewDataApp to
        autofill the annotation input form.

        There will be one button corresponding to each component if its name is used in any call to self.add_autofill().
        If the user clicks that button, the app will fill the annotation inputs with the specified values or source
        of data.

        Parameters
        ----------
        autofill_button_name: str
            Name of the button for autofilling.
        fill_value: Union[State, str, float]
            The value to fill the corresponding annotation input specified in annot_col.
            - State(id, attribute): It will fill with whatever value is currently in the layout and its
                corresponding attribute ('children', 'value')
            - str, float: Just a default value to fill if the data in the current component is used to autofill.
                Ex. Component A computes results with one method, while Component B computes results with another.
                You want to annotate with method you used to produce the results that fill the annotation inputs.
        annot_col: str
            The name of an annotation in self.review_data.data.annot_col_config_dict

        """
        if autofill_button_name not in self.autofill_dict.keys():
            self.autofill_dict[autofill_button_name] = {annot_col: fill_value}
        else:
            self.autofill_dict[autofill_button_name][annot_col] = fill_value

        # verify 
        self.app.gen_autofill_buttons_and_states(self.review_data_interface, self.autofill_dict)

    def run(self,
            review_data_table_df: pd.DataFrame = None,
            review_data_table_page_size: int = 5,
            collapsable=True,
            mode='external', 
            host='0.0.0.0', 
            port=8050):
        """
        Runs the app

        Parameters
        ----------
        review_data_table_df: dataframe with index that matches the index of the reviewer data object's index
        review_data_table_page_size: number of subjects to view
        mode: {'inline', 'external'}, default='external'
        host: str, default='0.0.0.0'
            Host address
        port: int, default=8050
            Port number

        """
        self.app.run(review_data=self.review_data_interface,
                     autofill_dict=self.autofill_dict,
                     annot_app_display_types_dict=self.annot_app_display_types_dict,
                     review_data_table_df=review_data_table_df,
                     review_data_table_page_size=review_data_table_page_size,
                     collapsable=collapsable,
                     mode=mode,
                     host=host,
                     port=port)

    def get_data_attribute(self, attribute: str):
        return getattr(self.review_data_interface.data, attribute)

    def list_data_attributes(self):
        return vars(self.review_data_interface.data).keys()

    def get_annot(self):
        return self.get_data_attribute('annot_df')

    def get_history(self):
        return self.get_data_attribute('history_df')
