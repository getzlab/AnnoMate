import pandas as pd
import numpy as np
import pathlib
import os
from IPython.display import display
from datetime import datetime, timedelta
import time

import plotly.express as px
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import Dash, dash_table
import dash
import dash_bootstrap_components as dbc
import functools
import inspect
import enum

from .ReviewData import ReviewData, ReviewDataAnnotation, AnnotationType
    

class AppComponent:
    
    def __init__(self, 
                 name, 
                 components, 
                 callback_output=[], 
                 callback_input=[],
                 callback_state=[],
                 new_data_callback=None, # update everything
                 internal_callback=None, # internal changes
                ):
        
        all_ids = np.array(get_component_ids(components))
        check_duplicates(all_ids, 'component')
        
        callback_output_ids = get_callback_io_ids(callback_output)
        check_duplicates(callback_output_ids, 'callback_output')
        check_callback_io_id_in_list(callback_output_ids, all_ids, ids_type='output_ids', all_ids_type='component_ids')
        callback_input_ids = get_callback_io_ids(callback_input)
        callback_state_ids = get_callback_io_ids(callback_state)
        callback_input_state_ids = callback_input_ids + callback_state_ids
        check_duplicates(callback_input_state_ids, 'callback_input and callback_state')
        check_callback_io_id_in_list(callback_input_state_ids, all_ids, ids_type='input_state_ids', all_ids_type='component_ids')
        
        if internal_callback is not None and inspect.signature(new_data_callback) != inspect.signature(internal_callback):
            raise ValueError(f'new_data_callback and internal_callback do not have the same signature.\n'
                             f'new_data_callback signature:{inspect.signature(new_data_callback)}\n'
                             f'internal_callback signature:{inspect.signature(internal_callback)}'
                            )
            
        self.name = name
        self.component = html.Div(components)
        self.all_component_ids = all_ids
        self.callback_output = callback_output
        self.callback_input = callback_input
        self.callback_state = callback_state
        self.new_data_callback = new_data_callback
        self.internal_callback = internal_callback

        
    
class ReviewDataApp:
    def __init__(self, review_data: ReviewData, host='0.0.0.0', port=8051):
        self.prop = None
        self.autofill_buttons = []
        self.autofill_input_dict = {} #{buttonid: {annot_col: Input(compoennt value)}}
        self.more_components = []  # TODO: set custom layout?
        self.review_data = review_data
        self.host = host
        self.port = port
        
        self.reviewed_data_df = pd.DataFrame(index=self.review_data.annot.index, columns=['label'])
        self.reviewed_data_df['label'] = self.reviewed_data_df.apply(self.gen_dropdown_labels, axis=1)
        self.reviewed_data_df.index.name = 'value'
        
    def gen_dropdown_labels(self, r):
        data_history_df = self.review_data.history[self.review_data.history["index"] == r.name]
        if not data_history_df.empty:
            return r.name + f' (Last update: {data_history_df["timestamp"].tolist()[-1]})'
        else:
            return r.name
        
        
    def run(self, mode, host='0.0.0.0', port=8050):
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = self.gen_layout()

        @app.callback(output=dict(history_table=Output(f'APP-history-table', 'children'),
                                  annot_panel=self.annotation_panel_component.callback_output,
                                  dropdown_list_options=Output(f'APP-dropdown-data-state', 'options'),
                                  more_component_outputs={c.name: c.callback_output for c in self.more_components}
                             ), 
                      inputs=dict(dropdown_value=Input('APP-dropdown-data-state', 'value'), 
                                  autofill_buttons=[Input(b.id, 'n_clicks') for b in self.autofill_buttons],
                                  autofill_inputs=self.autofill_input_dict,
                                  submit_annot_button=Input('APP-submit-button-state', 'n_clicks'),
                                  annot_input_state=self.annotation_panel_component.callback_state, #{annot.name: State(f"APP-{annot.name}-{annot.annot_type}-input-state", "value") for annot in self.review_data.annotate_data},
                                  more_component_inputs={c.name: c.callback_input for c in self.more_components}
                                 )
                     ) # TODO: add back more components
        def component_callback(dropdown_value, 
                               autofill_buttons,
                               autofill_inputs,
                               submit_annot_button, 
                               annot_input_state, 
                               more_component_inputs):
            
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            else:
                prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            output_dict = {'history_table': dash.no_update, 
                           'annot_panel': {annot_col: dash.no_update for annot_col in self.review_data.annot.columns}, 
                           'dropdown_list_options': dash.no_update,
                           'more_component_outputs': {c.name: [dash.no_update for i in range(len(c.callback_output))] for c in self.more_components}}

            if prop_id == 'APP-dropdown-data-state':
                for i in range(len(self.more_components)):
                    component = self.more_components[i]
                    # reset vs row dependent
                    component_output = component.new_data_callback(self.review_data.data, # Require call backs first two args be the dataframe and the index value
                                                          dropdown_value, 
                                                          *more_component_inputs[component.name])
                    output_dict['more_component_outputs'][component.name] = component_output
                    
                output_dict['history_table'] = dbc.Table.from_dataframe(self.review_data.history.loc[self.review_data.history['index'] == dropdown_value])
                output_dict['annot_panel'] = {annot_col: '' for annot_col in self.review_data.annot.columns} # TODO set defaults?
                            
            elif (prop_id == 'APP-submit-button-state') & (submit_annot_button > 0):
                self.review_data._update(dropdown_value, annot_input_state)
                output_dict['history_table'] = dbc.Table.from_dataframe(self.review_data.history.loc[self.review_data.history['index'] == dropdown_value])
                self.reviewed_data_df.loc[dropdown_value, 'label'] = self.gen_dropdown_labels(self.reviewed_data_df.loc[dropdown_value])
                output_dict['dropdown_list_options'] = self.reviewed_data_df.reset_index().to_dict('records')
            elif 'APP-autofill-' in prop_id:
                component_name = prop_id.split('APP-autofill-')[-1]
                for autofill_annot_col, value in autofill_inputs[prop_id].items():
                    output_dict['annot_panel'][autofill_annot_col] = value
            else:
                # identify component that changed and which outputs are changed
                for i in range(len(self.more_components)):
                    component = self.more_components[i]
                    if sum([c.component_id == prop_id for c in self.more_components[i].callback_input]) > 0:
                        try:
                            component_output = component.internal_callback(self.review_data.data, # Require call backs first two args be the dataframe and the index value
                                                              dropdown_value, 
                                                              *more_component_inputs[component.name])
                        except TypeError: # component.internal_callback is nonetype
                            component_output = component.new_data_callback(self.review_data.data, 
                                                                           dropdown_value, 
                                                                           *more_component_inputs[component.name])
                        output_dict['more_component_outputs'][component.name] = component_output # force having output as array specify names in the callback outputs? Or do it by dictionary
                pass
            return output_dict
        
        app.run_server(mode=mode, host=host, port=port, debug=True) 
        
        
    def gen_annotation_panel_component(self):
        annotation_data = self.review_data.review_data_annotation_list
        
        
        submit_annot_button = html.Button(id='APP-submit-button-state', n_clicks=0, children='Submit')
        
        # history panel
        
        def annotation_input(annot_name, annot: ReviewDataAnnotation):
            
            input_component_id = f"APP-{annot_name}-{annot.annot_type}-input-state"
            
            if annot.annot_type == AnnotationType.TEXTAREA.value:
                input_component = dbc.Textarea(size="lg", 
                                               id=input_component_id,
                                               value=annot.default,
                                              ), 
            elif annot.annot_type ==  AnnotationType.TEXT.value:
                input_component = dbc.Input(type="text", 
                                    id=input_component_id, 
                                    placeholder=f"Enter {annot_name}",
                                    value=annot.default,
                                   )
            elif annot.annot_type == AnnotationType.NUMBER.value:
                input_component = dbc.Input(type="number", 
                                    id=input_component_id, 
                                    placeholder=f"Enter {annot_name}",
                                    value=annot.default,
                                   )
            elif annot.annot_type == AnnotationType.CHECKLIST.value:
                input_component = dbc.Checklist(options=[{"label": f, "value": f} for f in annot.options],
                                                id=input_component_id, 
                                                value=annot.default),
            elif annot.annot_type == AnnotationType.RADIOITEM.value:
                input_component = dbc.RadioItems(
                                                options=[{"label": f, "value": f} for f in annot.options],
                                                value=annot.default,
                                                id=input_component_id,
                                            ),
            else:
                raise ValueError(f'Invalid annotation type "{annot.annot_type}"')
                
            return dbc.Row([dbc.Label(annot_name, html_for=input_component_id, width=2), dbc.Col(input_component)])
        
        panel_components = self.autofill_buttons + [annotation_input(annot.name, annot) for annot in self.review_data.review_data_annotation_list] + [submit_annot_button]
        panel_inputs = [Input('APP-submit-button-state', 'nclicks')]
        return AppComponent(name='APP-Panel',
                           components=panel_components, 
                           callback_output={annot.name: Output(f"APP-{annot.name}-{annot.annot_type}-input-state", "value") for annot in self.review_data.review_data_annotation_list},
                           callback_input=panel_inputs,
                           callback_state={annot.name: State(f"APP-{annot.name}-{annot.annot_type}-input-state", "value") for annot in self.review_data.review_data_annotation_list}
                          )
        
    def gen_layout(self):
        dropdown = html.Div(dcc.Dropdown(options=self.reviewed_data_df.reset_index().to_dict('records'), 
                                         value=None, 
                                         id='APP-dropdown-data-state'))
        
        self.dropdown_component = AppComponent(name='APP-dropdown-component',
                                               components=[dropdown])
        
        history_table = html.Div([dbc.Table.from_dataframe(pd.DataFrame(columns=self.review_data.history.columns))], id='APP-history-table')
        self.history_component = AppComponent(name='APP-history-component',
                                               components=[history_table])
        
        self.annotation_panel_component = self.gen_annotation_panel_component()
        
        layout = html.Div([dbc.Row(self.dropdown_component.component, justify='end'),
                           dbc.Row([dbc.Col(self.annotation_panel_component.component),
                                    dbc.Col(self.history_component.component)
                                   ]),
                           dbc.Row([dbc.Row(c.component) for c in self.more_components])
                          ])
        
        all_ids = np.array(get_component_ids(layout))
        check_duplicates(all_ids, 'full component')

        return layout
    
    def add_table_from_path(self, table_name, component_name, col, table_cols):
        
        table = html.Div(dbc.Table.from_dataframe(pd.DataFrame()), 
                                   id=component_name)
        table_component = AppComponent(component_name, 
                                       [html.H1(table_name), table], 
                                       new_data_callback = lambda df, idx: [dbc.Table.from_dataframe(pd.read_csv(df.loc[idx, col], sep='\t', encoding='iso-8859-1')[table_cols])],
                                       callback_output=[Output(component_name, 'children')], 
                                      )
        self.more_components.append(table_component)
        
    def add_custom_component(self, 
                             component_name, 
                             component_layout,
                             new_data_callback, # switching data
                             callback_output: [Output], 
                             callback_input: [Input]=[], 
                             internal_callback=None, # within the component
                             add_autofill=False,
                             autofill_dict={}, # annot_col: component output id
                             **kwargs):
        
        if add_autofill:
            autofill_button_component = html.Button(f'Use current {component_name} solution', 
                                                    id=f'APP-autofill-{component_name}', 
                                                    n_clicks=0)
            self.autofill_buttons += [autofill_button_component]
            self.autofill_input_dict[autofill_button_component.id] = autofill_dict
            
        component = AppComponent(component_name, 
                                 component_layout, 
                                 new_data_callback=lambda *args: new_data_callback(*args, **kwargs),
                                 internal_callback=lambda *args: internal_callback(*args, **kwargs),
                                 callback_output=callback_output, 
                                 callback_input=callback_input,
                                )
        
        ids_with_reserved_prefix_list = np.array(['APP-' in i for i in component.all_component_ids])
        if ids_with_reserved_prefix_list.any():
            raise ValueError(f'Some ids use reserved keyword "APP-". Please change the id name\n'
                             f'Invalid component ids: {component.all_component_ids[np.argwhere(ids_with_reserved_prefix_list).flatten()]}')
        
        self.more_components.append(component)
        all_ids = get_component_ids([c.component for c in self.more_components])
        check_duplicates(all_ids, f'ids found in previously added component from component named {component.name}')
        
        
# Validation

def get_component_ids(component):
    if isinstance(component, list) or isinstance(component, tuple):
        id_list = []
        for comp in component:
            id_list += get_component_ids(comp)
        return id_list
    elif isinstance(component, str):
        return []
    elif 'children' in component.__dict__.keys(): #isinstance(component, html.Div): # TODO: include dbc rows and columns
        children_ids = get_component_ids(component.children)
        if 'id' in component.__dict__.keys():
            children_ids += [component.id]
        return children_ids
    else:
        return [component.id] if 'id' in component.__dict__.keys() else []

def get_callback_io_ids(io_list):
    # TODO: might be a dictionary
    if isinstance(io_list, list):
        return [item.component_id for item in io_list]
    elif isinstance(io_list, dict):
        return [item.component_id for k, item in io_list.items()]

def check_callback_io_id_in_list(ids, all_ids, ids_type:str, all_ids_type:str):
    ids_not_in_all_ids = np.array(ids)[np.argwhere(np.array([id_name not in all_ids for id_name in ids])).flatten()]
    if len(ids_not_in_all_ids) > 0:
        raise ValueError(f"{ids_type} ids not in {all_ids_type}:\n"
                         f"{ids_type} ids: {ids_not_in_all_ids}\n"
                         f"{all_ids_type} ids: {all_ids}"
                        )

def check_duplicates(a_list, list_type: str):
    values, counts = np.unique(a_list, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"Duplicate {list_type}: {values[np.argwhere(counts > 1)].flatten()}")
            