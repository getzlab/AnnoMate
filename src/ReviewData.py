
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

from enum import Enum
class AnnotationType(Enum):
    TEXT = 'text'
    TEXTAREA = 'textarea'
    NUMBER = 'number'
    CHECKLIST = 'checklist'
    RADIOITEM = 'radioitem'

class ReviewDataAnnotation:
    
    def __init__(self, 
                 annot_type: AnnotationType, 
                 options: []=[], 
                 validate_input=None,
                 default=None
                ):
        '''
        validate_input: a custom function to verify input. Returns a boolean
        '''
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
                 review_dir: str, # path to directory to save info
                 df: pd.DataFrame, # optional if directory above already exists. 
                 annotate_data: {str: ReviewDataAnnotation}, # dictionary naming column and type of data (text, float, checkbox, radio)
                ):
        # check df index
        
        annotate_cols = list(annotate_data.keys())
        self.annotate_data = annotate_data
        
        self.review_dir = review_dir
        self.data_fn = f'{review_dir}/data.tsv'
        self.annot_fn = f'{review_dir}/annot.tsv'
        self.history_fn = f'{review_dir}/history.tsv'
        
        if not os.path.isdir(self.review_dir):
            os.mkdir(self.review_dir)
            self.data = df
            self.data.to_csv(self.data_fn, sep='\t')
            self.annot = pd.DataFrame(index=df.index, columns=annotate_cols) # Add more columns. If updating an existing column, will make a new one
            self.annot.to_csv(self.annot_fn, sep='\t')
            self.history = pd.DataFrame(columns=annotate_cols + ['index', 'timestamp']) # track all the manual changes, including time stamp
            self.history.to_csv(self.history_fn, sep='\t')
        else:
            self.data = pd.read_csv(self.data_fn, sep='\t', index_col=0)
            self.annot = pd.read_csv(self.annot_fn, sep='\t', index_col=0)
            self.history = pd.read_csv(self.history_fn, sep='\t', index_col=0)
            
        # Add additional annotation columns
        new_annot_cols = [c for c in annotate_cols if c not in self.annot.columns]
        self.annot[new_annot_cols] = np.nan
        
        for annot_name, annot in self.annotate_data.items():
            if annot.annot_type in [AnnotationType.CHECKLIST, AnnotationType.RADIOITEM]:
                self.annot[annot_name] = self.annot[annot_name].astype(object)
        
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
        
    def _update(self, data_idx, series):
        self.annot.loc[data_idx, list(series.keys())] = list(series.values())
        series['timestamp'] = datetime.today()
        series['index'] = data_idx
        self.history = self.history.append(series, ignore_index=True)
        
        # write to file
        self.data.to_csv(self.data_fn, sep='\t')
        self.annot.to_csv(self.annot_fn, sep='\t')
        self.history.to_csv(self.history_fn, sep='\t')
        
        