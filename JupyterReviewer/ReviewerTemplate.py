from .ReviewData import ReviewData, ReviewDataAnnotation
from .ReviewDataApp import ReviewDataApp

import pandas as pd
import numpy as np
import functools
import time
import typing
import os

from abc import ABC, abstractmethod


# class ReviewerTemplate(ABC):
    
class ReviewerTemplate(ABC):
    
    def __init__(self):
        self.review_data = None
        self.app = None
        self.autofill_dict = {}    
    
    @abstractmethod
    def gen_review_data(self,
                        review_data_fn: str, 
                        description: str='', 
                        df: pd.DataFrame = pd.DataFrame(), 
                        review_data_annotation_dict: {str:ReviewDataAnnotation} = {}, 
                        reuse_existing_review_data_fn: str = None, *args, **kwargs) -> ReviewData:
        
        return None
        
        
    @abstractmethod
    def gen_review_app(self) -> ReviewDataApp:
        app = ReviewDataApp()
        app.add_component()
        
        return app
        
    @abstractmethod
    def gen_autofill(self):
        return None
    
    # Public methods
    def set_review_data(self,
                        review_data_fn: str, 
                        description: str='', 
                        df: pd.DataFrame = pd.DataFrame(), 
                        review_data_annotation_dict: {str: ReviewDataAnnotation} = {}, 
                        reuse_existing_review_data_fn: str = None,  
                        **kwargs):
        
        if os.path.exists(review_data_fn) or ((reuse_existing_review_data_fn is not None) and 
                                              os.path.exists(reuse_existing_review_data_fn)):
            self.review_data = ReviewData(review_data_fn=review_data_fn,
                                          description=description,
                                          df=df,
                                          review_data_annotation_dict=review_data_annotation_dict,
                                          reuse_existing_review_data_fn=reuse_existing_review_data_fn)
        else:
            self.review_data = self.gen_review_data(review_data_fn,
                                               description,
                                               df,
                                               # review_data_annotation_list,
                                               reuse_existing_review_data_fn,
                                               **kwargs)
    
    def set_review_app(self, *args, **kwargs):
        self.app = self.gen_review_app(*args, **kwargs)
        self.gen_autofill()
        
    def add_autofill(self, component_name: str, autofill_dict: dict):
        self.autofill_dict[component_name] = autofill_dict
        
        # verify 
        self.app.gen_autofill_buttons_and_states(self.review_data, self.autofill_dict)
        
    def run(self, 
            mode='external', 
            host='0.0.0.0', 
            port=8050):
        self.app.run(review_data=self.review_data, 
                     autofill_dict=self.autofill_dict,
                     mode=mode,
                     host=host,
                     port=port)
    
    
    