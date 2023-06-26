from AnnoMate.Data import Data
import pandas as pd
from typing import Dict


class PatientSampleData(Data):

    def __init__(self,
                 index,
                 description,
                 participant_df: pd.DataFrame,
                 sample_df: pd.DataFrame,
                 annot_df: pd.DataFrame = None,
                 annot_col_config_dict: Dict = None,
                 history_df: pd.DataFrame = None,
                 ):
        super().__init__(index=index,
                         description=description,
                         annot_df=annot_df,
                         annot_col_config_dict=annot_col_config_dict,
                         history_df=history_df)

        self.sample_df = sample_df
        self.participant_df = participant_df

        # check tables are linked, other validation
        self.participant_col_in_sample = 'participant'
