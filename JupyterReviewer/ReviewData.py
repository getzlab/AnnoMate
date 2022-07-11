import pandas as pd
from datetime import datetime
import os
import numpy as np
import warnings
import pickle
from typing import List, Dict

valid_annotation_types = ["multi", "float", "int", "string"]


class ReviewDataAnnotation:
    
    def __init__(self,
                 annot_value_type: str,
                 options: List = None,
                 validate_input=None,
                 default=None
                ):
        """
        annot_value_type: value for the annotation type. Determines dtype of the dataframe column
        options:          list of values inputs are allowed to be. For CHECKLIST, RADIOITEM, and DROPDOWN
        validate_input:   a custom function to verify annotation input. Takes a single input and returns a boolean
        """

        if annot_value_type not in valid_annotation_types:
            raise ValueError(
                f'annot_type {annot_value_type} is not a valid annotation value type. '
                f'Valid types are {valid_annotation_types}')

        self.annot_value_type = annot_value_type
        self.options = options
        self.validate_input = validate_input
        self.default = default
        
    def validate(self, x):
        if self.options is not None:
            for item in np.array(x).flatten():
                if item not in self.options and item != '':
                    raise ValueError(f'Input {item} is not in the specified options {self.options}')
                
        if self.validate_input is not None:
            if not self.validate_input(x):
                raise ValueError(f'Input {x} is invalid')
        
    def __str__(self):
        return str(self.__dict__)


class ReviewData:
    
    def __init__(self, 
                 review_data_fn: str, 
                 description: str='', 
                 df: pd.DataFrame = pd.DataFrame(), 
                 # review_data_annotations_list: List[ReviewDataAnnotation] = None,
                 reuse_existing_review_data_fn: str = None, 
                ):
        """
        review_data_fn:                path to save review data object
        
        description:                   describe the review session. This is useful if you copy the history of this 
                                       object to a new review data object
                                       
        df:                            pandas dataframe with the data to review
        
        # review_data_annotations_list:  List of ReviewDataAnnotation objects to define the ReviewData.annot table
        
        reuse_existing_review_data_fn: path to existing review data object to copy
        """
        if df.index.shape[0] != df.index.unique().shape[0]:
            raise ValueError(f'Input dataframe df does not have unique index values.')
        
        self.review_data_annotation_dict = {}
        
        if reuse_existing_review_data_fn == review_data_fn:
            raise ValueError(f'Inputs for review_data_fn and reuse_existing_review_data_fn are the same. '
                             'Pass in a different file name for reuse_existing_review_data_fn\n'
                             f'review_data_fn: {review_data_fn}\n'
                             f'reuse_existing_review_data_fn: {reuse_existing_review_data_fn}')
        
        if not os.path.exists(review_data_fn):
            if reuse_existing_review_data_fn is None:
                # annotate_cols = [ann.annot_name for ann in review_data_annotations_list]
                # self.annot = pd.DataFrame(index=df.index, columns=annotate_cols) # Add more columns. If updating an existing column, will make a new one
                self.annot = pd.DataFrame(index=df.index)
                # self.history = pd.DataFrame(columns=['index', 'timestamp', 'review_data_fn'] + annotate_cols) # track all the manual changes, including timestamp
                self.history = pd.DataFrame(columns=['index', 'timestamp', 'review_data_fn'])
            else:
                print(f'Copying from existing review data session to new review data session...\n'
                      f'{reuse_existing_review_data_fn} --> {review_data_fn}')
                self.load(reuse_existing_review_data_fn)
                
                missing_df_indices = np.array([i not in self.annot.index for i in df.index])
                if missing_df_indices.any():
                    raise ValueError(f'df input contains indices that does not already exist in the previous '
                                     f'review session.\n '
                                     f'Unavailable indices: {df.loc[missing_df_indices].index.tolist()}')
                if df.index.shape[0] != self.data.shape[0]:
                    warnings.warn(f'df input has fewer indices ({df.index.shape[0]}) than the original '
                                  f'review session df input ({self.data.shape[0]}). '
                                  'New review session will only contain the previous data '
                                  'corresponding to newest df indices')

                self.annot = self.annot.loc[df.index]
                self.history = self.history.loc[self.history['index'].isin(df.index)]

            self.data = df  # overwrite data frame.
            self.review_data_fn = review_data_fn # change path to save object
            self.description = description
            # self._add_annotations(review_data_annotations_list)
            self.save()
        else:
            print(f'Loading existing review session {review_data_fn}...')
            self.load(review_data_fn)

        # Add additional columns to table
        if not df.equals(self.data):
            new_data_cols = [c for c in df.columns if c not in self.data.columns]
            not_new_data_cols = [c for c in df.columns if c in self.data.columns]
            self.data[new_data_cols] = df[new_data_cols]
            
            if not self.data[not_new_data_cols].equals(df[not_new_data_cols]):
                warnings.warn(f'Input data dataframe shares columns with existing data, but are not equal.\n' + 
                              f'Only adding columns {new_data_cols} to the ReviewData.data dataframe\n' + 
                              f'Remaining columns are not going to be updated.' + 
                              f'If you intend to change the ReviewData.data attribute, '
                              'make a new review data object and pass in this object\'s path to reuse_existing_'
                              'review_data_fn:\n\n'
                              f'new_rd = ReviewData(review_data_fn=new_fn, df=updated_df, '
                              f'reuse_existing_review_data_fn={review_data_fn})')
            
    def load(self, review_data_fn):
        f = open(review_data_fn, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

    def save(self):
        f = open(self.review_data_fn, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
    
    def pre_fill_annot(self, df: pd.DataFrame):
        """
        df: Dataframe with indices and columns in the ReviewData.annot table
        """
        valid_annot_cols = [c for c in df.columns if c in self.annot.columns]
        valid_data_idx = [data_idx for data_idx in df.index if data_idx in self.annot.index]
        for data_idx in valid_data_idx:
            self._update(data_idx, df.loc[data_idx, valid_annot_cols])
            
        invalid_annot_cols = [c for c in df.columns if c not in self.annot.columns]
        invalid_data_idx = [data_idx for data_idx in df.index if data_idx not in self.annot.index]
        warnings.warn(f'There was extra data in your input that was not added to the ReviewData object.\n' +
                      f'Invalid annotation cols: {invalid_annot_cols}\n' +
                      f'Invalid data indices: {invalid_data_idx}') 
        
    def add_annotation(self,
                       annot_name: str,
                       review_annot: ReviewDataAnnotation):
        """
        review_annot: a ReviewDataAnnotation to add to the review data object
        """
        self._add_annotations({annot_name: review_annot})
    
    def _add_annotations(self, review_data_annotation_dict: Dict):

        new_annot_data = {annot_name: ann for annot_name, ann in review_data_annotation_dict.items() if
                          annot_name not in self.review_data_annotation_dict.keys()}
        
        for name, ann in new_annot_data.items():
            self.review_data_annotation_dict[name] = ann

        self.annot[list(new_annot_data.keys())] = np.nan
        self.history[list(new_annot_data.keys())] = np.nan
        
        for name, annot_data in new_annot_data.items():
            if annot_data.annot_value_type == 'multi':
                self.annot[name] = self.annot[name].astype(object)
            elif annot_data.annot_value_type == 'float':
                self.annot[name] = self.annot[name].astype(float)
            # elif annot_data.annot_value_type == 'int':
            #     self.annot[name] = self.annot[name].astype(int)
            elif annot_data.annot_value_type == 'string':
                self.annot[name] = self.annot[name].astype(str)

        self.save()
        
    def _update(self, data_idx, dictionary):
        if list(self.annot.loc[data_idx, list(dictionary.keys())].values) != list(dictionary.values()):
            self.annot.loc[data_idx, list(dictionary.keys())] = list(dictionary.values())
            dictionary['timestamp'] = datetime.today()
            dictionary['index'] = data_idx
            dictionary['review_data_fn'] = self.review_data_fn
            self.history = pd.concat([self.history, pd.Series(dictionary).to_frame().T])
            self.save()
            
    def export(self, path: str):
        """
        path: local or gsurl path to directory to save object's dataframe objects
        """
        self.data.to_csv(f'{path}/data.tsv', sep='\t')
        self.annot.to_csv(f'{path}/annot.tsv', sep='\t')
        self.history.to_csv(f'{path}/history.tsv', sep='\t')
        