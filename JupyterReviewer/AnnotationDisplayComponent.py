from abc import ABC, abstractmethod
from .Data import DataAnnotation, valid_annotation_types
import dash_bootstrap_components as dbc
from dash import dcc
import numpy as np
from typing import List

class AnnotationDisplayComponent(ABC):
    
    def __init__(self, default_display_value=None, display_output_format=None, **kwargs):
        self.display_output_format = display_output_format
        self.default_display_value = default_display_value
        self.default_compatible_types = valid_annotation_types
        
    @abstractmethod
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        pass
    
    def __str__(self):
        return self.__class__.__name__
    
class MultiValueAnnotationDisplayComponent(AnnotationDisplayComponent):
    
    def __init__(self, default_display_value=None, display_output_format=None):
        super().__init__(default_display_value=default_display_value, display_output_format=display_output_format)
        self.default_compatible_types = ['multi']
        
    @abstractmethod
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        pass
    
    
class TextAnnotationDisplay(AnnotationDisplayComponent):
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Input(
            type="text",
            id=component_id,
            value=self.default_display_value,
        )
    
class TextAreaAnnotationDisplay(AnnotationDisplayComponent):
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Textarea(
            size="lg", 
            id=component_id,
            value=self.default_display_value
        )
    
class NumberAnnotationDisplay(AnnotationDisplayComponent):
    def __init__(self, default_display_value=None, display_output_format=None, empty_values: List =None):
        super().__init__(default_display_value=default_display_value, display_output_format=display_output_format)
        self.default_compatible_types = ['float', 'int']

        self.empty_values = ['', None, 'None', 'NaN'] if empty_values is None else empty_values
        if self.display_output_format is None:
            self.display_output_format = lambda x: np.nan if x in self.empty_values else x
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Input(
            type="number",
            id=component_id,
            value=self.default_display_value,
        )
    
class ChecklistAnnotationDisplay(MultiValueAnnotationDisplayComponent):
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Checklist(
            options=[{"label": f, "value": f} for f in data_annot.options],
            id=component_id, 
            value='' if self.default_display_value is None else self.default_display_value # Checklist does not take NoneType values
        ) 
    
class RadioitemAnnotationDisplay(AnnotationDisplayComponent):
    
    def __init__(self, default_display_value=None, display_output_format=None):
        super().__init__(default_display_value=default_display_value, display_output_format=display_output_format)
        self.default_compatible_types = ['string']
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.RadioItems(
            options=[{"label": f, "value": f} for f in data_annot.options],
            value=self.default_display_value,
            id=component_id,
        )
    
class SelectAnnotationDisplay(AnnotationDisplayComponent):
    
    def __init__(self, default_display_value=None, display_output_format=None):
        super().__init__(default_display_value=default_display_value, display_output_format=display_output_format)
        self.default_compatible_types = ['string']
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Select(
            options=[{"label": f, "value": f} for f in data_annot.options],
            value=self.default_display_value,
            id=component_id,
        )


class MultiValueSelectAnnotationDisplay(MultiValueAnnotationDisplayComponent):
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        
        return dcc.Dropdown(
            options=[{"label": f, "value": f} for f in data_annot.options],
            multi=True,
            id=component_id,
            value=self.default_display_value 
        )
        