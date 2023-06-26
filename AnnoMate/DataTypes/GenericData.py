from AnnoMate.Data import Data
import pandas as pd
from typing import Dict


class GenericData(Data):

    def __init__(self,
                 index,
                 description,
                 df: pd.DataFrame,
                 annot_df: pd.DataFrame = None,
                 annot_col_config_dict: Dict = None,
                 history_df: pd.DataFrame = None,
                 ):
        """
        A basic Data object with only one dataframe. This is useful if you are reviewing a
        single data type and all the data type (ie just sample data, just patient data, etc.) and/or
        all the data you want to review can be contained in a single dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            A pandas dataframe. The index should correspond to the data to review.
        """
        super().__init__(index=index,
                         description=description,
                         annot_df=annot_df,
                         annot_col_config_dict=annot_col_config_dict,
                         history_df=history_df)

        self.df = df
