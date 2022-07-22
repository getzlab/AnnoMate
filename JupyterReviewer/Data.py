from typing import List, Dict
import pandas as pd
from abc import ABC, abstractmethod

class Data:

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