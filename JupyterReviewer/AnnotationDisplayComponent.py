from abc import ABC, abstractmethod
from .Data import DataAnnotation, valid_annotation_types
import dash_bootstrap_components as dbc

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
    
    def __init__(self, default_display_value=None, display_output_format=None):
        super().__init__(default_display_value=default_display_value, display_output_format=display_output_format)
        self.default_compatible_types = ['number']
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Input(
            type="number",
            id=component_id,
            value=self.default_display_value,
        )
    
class ChecklistAnnotationDisplay(AnnotationDisplayComponent):
    
    def gen_input_component(self, data_annot: DataAnnotation, component_id: str):
        return dbc.Checklist(
            options=[{"label": f, "value": f} for f in data_annot.options],
            id=component_id, 
            value='' if self.default_display_value is None else self.default_display_value # Checklist does not take NoneType values
        ) 
    
class RadioitemAnnotationDisplay(AnnotationDisplayComponent):
    
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
