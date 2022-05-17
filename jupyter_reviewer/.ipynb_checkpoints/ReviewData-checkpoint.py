
import pandas as pd
import pathlib
import os
from IPython.display import display
from datetime import datetime, timedelta
import time


import pandas as pd
from datetime import datetime
import os
import numpy as np
import warnings
import pickle

from enum import Enum
class AnnotationType(Enum):
    TEXT = 'text'
    TEXTAREA = 'textarea'
    NUMBER = 'number'
    CHECKLIST = 'checklist'
    RADIOITEM = 'radioitem'

class ReviewDataAnnotation:
    
    def __init__(self,
                 name,
                 annot_type: AnnotationType, 
                 options: []=[], 
                 validate_input=None,
                 default=None
                ):
        '''
        validate_input: a custom function to verify input. Returns a boolean
        '''
        self.name = name
        self.annot_type = annot_type
        self.options = options
        self.validate_input = validate_input
        self.default = default
        
    def validate(self, x):
        if len(self.options) > 0:
            for item in np.array([x]).flatten():
                if item not in self.options:
                    raise ValueError(f'Input {item} is not in the specified options {self.options} for annotation named {self.name}')
                
        if self.validate_input is not None:
            if not self.validate_input(x):
                raise ValueError(f'Input {x} is invalid for annotation {self.name}. Check validate_input method')
        

class ReviewData:
    
    def __init__(self, 
                 review_data_fn: str, # path to save object
                 df: pd.DataFrame, # optional if directory above already exists. 
                 review_data_annotation_list: [ReviewDataAnnotation], # list
                ):
        # check df index
        self.review_data_annotation_list = review_data_annotation_list
        
        self.review_data_fn = review_data_fn
        
        if not os.path.exists(self.review_data_fn):
            annotate_cols = [c.name for c in review_data_annotation_list]
            self.data = df
            self.annot = pd.DataFrame(index=df.index, columns=annotate_cols) # Add more columns. If updating an existing column, will make a new one
            self.history = pd.DataFrame(columns=annotate_cols + ['index', 'timestamp']) # track all the manual changes, including time stamp
            self.save()
        else:
            self.load()
            
#         # Add additional annotation columns
#         new_annot_cols = [c for c in annotate_cols if c not in self.annot.columns]
#         self.annot[new_annot_cols] = np.nan
#         self.history[new_annot_cols] = np.nan
        
#         for annot_name, annot in self.annotate_data.items():
#             if annot.annot_type in [AnnotationType.CHECKLIST, AnnotationType.RADIOITEM]:
#                 self.annot[annot_name] = self.annot[annot_name].astype(object)

        self._add_annotations(review_data_annotation_list)
        
        # Add additional columns to table
        if not df.equals(self.data):
            new_data_cols = [c for c in df.columns if c not in self.data.columns]
            not_new_data_cols = [c for c in df.columns if c in self.data.columns]
            self.data[new_data_cols] = df[new_data_cols]
            
            if not self.data[not_new_data_cols].equals(df[not_new_data_cols]):
                warnings.warn(f'Input data dataframe shares columns with existing data, but are not equal.\n' + 
                              f'Only adding columns {new_data_cols} to the ReviewData.data dataframe\n' + 
                              f'Remaining columns are not going to be updated.' + 
                              f'If you intend to change the ReviewData.data attribute, make a new session directory and prefill the annotation data')
            
    def load(self):
        f = open(self.review_data_fn, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 


    def save(self):
        f = open(self.review_data_fn, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
    
    
    def pre_fill_annot(df: pd.DataFrame):
        # TODO: check index already exists. Use _update()
        valid_annot_cols = [c for c in df.columns if c in self.annot.columns]
        valid_data_idx = [data_idx for data_idx in df.index if data_idx in self.annot.index]
        for data_idx in valid_data_idx:
            self._update(data_idx, r[valid_annot_cols])
            
        invalid_annot_cols = [c for c in df.columns if c not in self.annot.columns]
        invalid_data_idx = [data_idx for data_idx in df.index if data_idx not in self.annot.index]
        warnings.warn(f'There was extra data in your input that was not added to the ReviewData object.\n' +
                      f'Invalid annotation cols: {invalid_annot_cols}\n' +
                      f'Invalid data indices: {invalid_data_idx}') 
        
    def add_annotation(self, 
                       name,
                       annot_type: AnnotationType, 
                       options: []=[], 
                       validate_input=None,
                       default=None):
        
        review_annot = ReviewDataAnnotation(name=name, 
                                            annot_type=annot_type, 
                                            options=options, 
                                            validate_input=validate_input, 
                                            default=default)
        self._add_annotations([review_annot])
    
    def _add_annotations(self, review_data_annotation_list):
        
        # Add additional annotation columns
        new_annot_data_list = [c for c in review_data_annotation_list if c.name not in self.annot.columns]
        new_annot_cols = [c.name for c in new_annot_data_list]
        self.annot[new_annot_cols] = np.nan
        self.history[new_annot_cols] = np.nan
        self.review_data_annotation_list += new_annot_data_list
        
        for annot_data in new_annot_data_list:
            if annot_data.annot_type in [AnnotationType.CHECKLIST, AnnotationType.RADIOITEM]:
                self.annot[annot_data.name] = self.annot[annot_data.name].astype(object)
                
        
        
    def _update(self, data_idx, series):
        self.annot.loc[data_idx, list(series.keys())] = list(series.values())
        series['timestamp'] = datetime.today()
        series['index'] = data_idx
        self.history = self.history.append(series, ignore_index=True)
        self.save()
        
        