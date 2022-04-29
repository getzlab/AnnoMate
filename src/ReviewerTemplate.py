from .ReviewData import ReviewData, ReviewDataAnnotation
from .ReviewDataApp import ReviewDataApp

import pandas as pd
import numpy as np
import functools
import time

from abc import ABC, abstractmethod


class ReviewerTemplate(ABC):
    
    @abstractmethod
    def gen_review_data_object(session_dir, 
                               df: pd.DataFrame, 
                               more_annot_cols: {str: ReviewDataAnnotation}) -> ReviewData:
        pass
        
    
    @abstractmethod
    def gen_review_data_app(review_data_obj: ReviewData, *args, **kwargs) -> ReviewDataApp:
        pass
    
    