from ReviewData import ReviewData, ReviewDataAnnotation
from ReviewDataApp import ReviewDataApp

import pandas as pd
import numpy as np
import functools
import time
import typing
import os

from abc import ABC, abstractmethod


class ReviewerTemplate(ABC):
    
    @abstractmethod
    def gen_review_data_object(review_data_obj_fn: typing.Union[str, bytes, os.PathLike], 
                               df: pd.DataFrame, 
                               more_annot_cols: [ReviewDataAnnotation]) -> ReviewData:
        """
        review_data_obj_fn: path to save review data object
        df:                 pandas dataframe with the data to review
        more_annot_cols:    additional annotation columns a user may want to include (id [ReviewDataAnnotation(), ...])
        """
        pass
        
    
    @abstractmethod
    def gen_review_data_app(review_data_obj: ReviewData, *args, **kwargs) -> ReviewDataApp:
        pass
    
    