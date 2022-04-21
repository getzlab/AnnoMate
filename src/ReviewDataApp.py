import pandas as pd
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

from src.ReviewData import ReviewData, ReviewDataAnnotation, AnnotationType

class AppComponent:
    
    def __init__(self, name, 
                 components, 
                 callback=None, 
                 callback_output=[], 
                 callback_input=[],
                 callback_state=[],
                ):
        self.name = name
        self.component = html.Div(components)
        self.callback = callback
        self.callback_output = callback_output
        self.callback_input = callback_input
        self.callback_state = callback_state
        
        # TODO: option to update anotations
        # TODO: reset function (switching samples) and a page function
    
class ReviewDataApp:
    def __init__(self, review_data: ReviewData, host='0.0.0.0', port=8051):
        self.prop = None
        self.autofill_buttons = []
        self.autofill_input_dict = {} #{buttonid: {annot_col: Input(compoennt value)}}
        self.more_components = []  # TODO: set custom layout?
        self.review_data = review_data
        self.host = host
        self.port = port
        
        # check component ids are not duplicated
        
    def run_app(self, mode, host='0.0.0.0', port=8050):
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = self.gen_layout()

        @app.callback(output=dict(history_table=Output(f'APP-history-table', 'children'),
                                  annot_panel=self.annotation_panel_component.callback_output,
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
                           'more_component_outputs': {c.name: [dash.no_update for i in range(len(c.callback_output))] for c in self.more_components}}
            
            if prop_id == 'APP-dropdown-data-state':

                for i in range(len(self.more_components)):
                    component = self.more_components[i]
                    # reset vs row dependent
                    component_output = component.callback(self.review_data.data.loc[dropdown_value], *more_component_inputs[component.name])
                    output_dict['more_component_outputs'][component.name] = component_output # force this? specify names in the callback outputs?
                    
                output_dict['history_table'] = dbc.Table.from_dataframe(self.review_data.history.loc[self.review_data.history['index'] == dropdown_value])
                output_dict['annot_panel'] = {annot_col: '' for annot_col in self.review_data.annot.columns} # TODO set defaults?
                            
            elif (prop_id == 'APP-submit-button-state') & (submit_annot_button > 0):
                self.review_data._update(dropdown_value, annot_input_state)
                output_dict['history_table'] = dbc.Table.from_dataframe(self.review_data.history.loc[self.review_data.history['index'] == dropdown_value])
            elif 'APP-autofill-' in prop_id:
                component_name = prop_id.split('APP-autofill-')[-1]
                for autofill_annot_col, value in autofill_inputs[prop_id].items():
                    output_dict['annot_panel'][autofill_annot_col] = value
            else:
                # identify component that changed and which outputs are changed
                for i in range(len(self.more_components)):
                    component = self.more_components[i]
                    if sum([c.component_id == prop_id for c in self.more_components[i].callback_input]) > 0:
                        component_output = component.callback(self.review_data.data.loc[dropdown_value], *more_component_inputs[component.name])
                        output_dict['more_component_outputs'][component.name] = component_output # force having output as array specify names in the callback outputs?
                pass
            return output_dict
        
        app.run_server(mode=mode, host=host, port=port, debug=True) 
        
        
    def gen_annotation_panel_component(self):
        annotation_data = self.review_data.annotate_data
        
        
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
                flags = np.arange(0, 1, 0.2)
                input_component = dbc.Checklist(options=[{"label": f, "value": f} for f in annot.options],
                                                id=input_component_id, 
                                                value=annot.default),
            elif annot.annot_type == AnnotationType.RADIOITEM.value:
                # TODO: how to add in options
                input_component = dbc.RadioItems(
                                                options=[{"label": f, "value": f} for f in annot.options],
                                                value=annot.default,
                                                id=input_component_id,
                                            ),
            else:
                raise ValueError(f'Invalid annotation type "{annot.annot_type}"')
                
            return dbc.Row([dbc.Label(annot_name, html_for=input_component_id, width=2), dbc.Col(input_component)])
        
        panel_components = self.autofill_buttons + [annotation_input(annot_name, annot) for annot_name, annot in self.review_data.annotate_data.items()] + [submit_annot_button]
        panel_inputs = [Input('APP-submit-button-state', 'nclicks')]
        return AppComponent(name='APP-Panel',
                           components=panel_components, 
                           callback_output={annot_name: Output(f"APP-{annot_name}-{annot.annot_type}-input-state", "value") for annot_name, annot in self.review_data.annotate_data.items()},
                           callback_input=panel_inputs,
                           callback_state={annot_name: State(f"APP-{annot_name}-{annot.annot_type}-input-state", "value") for annot_name, annot in self.review_data.annotate_data.items()}
                          )
        
    def gen_layout(self):
        
        dropdown = html.Div(dcc.Dropdown(options=self.review_data.data.index, 
                                         value=self.review_data.data.index[0], 
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

        return layout
    
    def add_table_from_path(self, table_name, component_name, col, table_cols):
        
        table = html.Div(dbc.Table.from_dataframe(pd.read_csv(self.review_data.data.iloc[0][col], sep='\t')[table_cols]), 
                                   id=component_name)
        table_component = AppComponent(component_name, 
                                       [html.H1(table_name), table], 
                                       lambda r: [dbc.Table.from_dataframe(pd.read_csv(r[col], sep='\t')[table_cols])],
                                       callback_output=[Output(component_name, 'children')], 
                                      )
        self.more_components.append(table_component)
        
    def add_custom_component(self, 
                             component_name, 
                             component_layout,
                             func, 
                             callback_output, 
                             callback_input=[], 
                             add_autofill=False,
                             autofill_dict={}, # annot_col: component output id
                             **kwargs):
        
        if add_autofill:
            autofill_button_component = html.Button(f'Use {component_name} solution', 
                                                    id=f'APP-autofill-{component_name}', 
                                                    n_clicks=0)
            self.autofill_buttons += [autofill_button_component]
            self.autofill_input_dict[autofill_button_component.id] = autofill_dict
            
        component = AppComponent(component_name, component_layout, 
                                  lambda *args: func(*args, **kwargs),
                                  callback_output=callback_output, 
                                  callback_input=callback_input
                                 )
        
        self.more_components.append(component)
        
        
        
    