import pandas as pd
import numpy as np
# from jupyter_dash import JupyterDash
from dash import jupyter_dash
from dash import dcc
from dash import html, Dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc
import inspect
from collections import OrderedDict
from typing import Dict
import copy
import warnings
from typing import Union, List, Tuple
from math import floor
from pathlib import Path
import os

from .ReviewDataInterface import ReviewDataInterface
from .Data import DataAnnotation, validate_annot_data
from .AnnotationDisplayComponent import AnnotationDisplayComponent

valid_annotation_app_display_types = ['text',
                                      'textarea',
                                      'number',
                                      'checklist',
                                      'radioitem',
                                      'select']

class AppComponent:
    
    def __init__(self, 
                 name: str, 
                 layout: Union[List, Tuple, html.Div],
                 callback_output: [Output] = [],
                 callback_input: [Input] = [],
                 callback_state: [State] = [],
                 callback_state_external: [State] = [],
                 new_data_callback=None,
                 internal_callback=None,
                 use_name_as_title=True):
        
        """
        Component in the plotly dashboard app. Each component is made up of a layout and callback functions
        defining interactive behaviors

        Parameters
        ----------
        name: str
            Unique name for the component. Will be used as the title displayed in the dashboard
            (preceeding the layout)
        
        layout: html
            dash html layout object (ex. html.Div([html.H1(children="heading", id='heading'), ...])),
        
        callback_output: List[Output]
            List of Output() objects pointing to objects in the layout
            (ex. [Output('heading', 'children')]) where callback function outputs will be stored
                          
        callback_input: List[Input]
            List of Input() objects pointing to objects in the layout
            (ex. [Input('heading', 'children')]). Changes to these specified Input objects
            will activate the callback function and its current value will be used as an input
                          
        callback_state: List[Input]
            List of State() objects pointing to objects in the layout
            (ex. [State('heading', 'children')]). It's value will be used as a parameter
            if ANY components' callback function is activated.

        callback_state_external: List[State]
            List of State() objects pointing to objects in the layout
            (ex. [State('heading', 'children')]) that do NOT exist in the layout,
            but instead from other AppComponent objects included in the same ReviewDataApp.
            Note that there will be no native check if the component id exists until runtime
            instantiation of the ReviewDataApp.
            It's value will be used as a parameter if the component's callback function
            is activated.
                          
        new_data_callback: func
            a function (defined separately or a lambda) that defines how the component will be updated
            when the ReviewData
            object switches to a new index or row to review. Requirements are:
            - The first two parameters will be the ReviewData.data object and the index of
                the current row to be reviewed.
            - The following parameters will be the inputs defined IN ORDER by the order of the
                Inputs() in callback_input
            - The next parameters will be the states defined IN ORDER by the order of the States()
                in callback_state
            - The next parameters will be the states defined IN ORDER by the order of the States()
                in callback_state_external
            - Any remaining parameters have keywords and are at the end
            - **new_data_callbackinternal_callback must have the same signature
                Example:
                    def myfunc(data: Data, idx, input1, input2, state1, **kwargs):...

                    callback_input = [Input('id-of-object-with-value-for-input1', 'value'),
                                      Input('id-of-object-with-value-for-input2', 'children')]

        internal_callback: func
            a function (defined separately or a lambda) that defines how the component will be updated
            when one of the defined callback_input Inputs are changed.
            Same requirements as new_data_callback

        use_name_as_title: bool
            use the `name` parameter as a title for the component.
        """
        
        all_ids = np.array(get_component_ids(layout))
        check_duplicates(all_ids, 'component')

        # preprocess callback_output to include allow_duplicate=True
        if isinstance(callback_output, list):
            new_callback_output = [
                Output(**{
                    'component_id': output.component_id, 
                    'component_property': output.component_property, 
                    'allow_duplicate': True
                }) for output in callback_output
            ]  
        elif isinstance(callback_output, dict):
            new_callback_output = {
                k: Output(**{
                    'component_id': output.component_id, 
                    'component_property': output.component_property, 
                    'allow_duplicate': True
                }) for k, output in callback_output.items()
            }
        else:
            raise ValueError(f'callback output {callback_output} is neither a list or a dictionary.')

        check_duplicate_objects(new_callback_output, 'callback_output_objects')
        callback_output_ids = get_callback_io_ids(new_callback_output, expected_io_type=Output)
        check_callback_io_id_in_list(callback_output_ids, all_ids, ids_type='output_ids', all_ids_type='component_ids')
        
        check_duplicate_objects(callback_input, 'callback_input_objects')
        callback_input_ids = get_callback_io_ids(callback_input, expected_io_type=Input)
        check_callback_io_id_in_list(callback_input_ids, all_ids, ids_type='input_ids', all_ids_type='component_ids')
        
        check_duplicate_objects(callback_state, 'callback_state_objects')
        callback_state_ids = get_callback_io_ids(callback_state, expected_io_type=State)
        check_callback_io_id_in_list(callback_state_ids, all_ids, ids_type='state_ids', all_ids_type='component_ids')

        if internal_callback is not None and \
                inspect.signature(new_data_callback) != inspect.signature(internal_callback):
            raise ValueError(f'new_data_callback and internal_callback do not have the same signature.\n'
                             f'new_data_callback signature:{inspect.signature(new_data_callback)}\n'
                             f'internal_callback signature:{inspect.signature(internal_callback)}')
            
        self.name = name
        if use_name_as_title:
            self.layout = html.Div([html.H1(name), html.Div(layout)])
        else:
            self.layout = html.Div(layout)
            
        self.all_component_ids = all_ids
        self.callback_output = new_callback_output
        self.callback_input = callback_input
        self.callback_state = callback_state
        self.callback_state_external = callback_state_external
        
        self.new_data_callback = new_data_callback
        self.internal_callback = internal_callback

    
class ReviewDataApp:
    def __init__(self):
        """
        Object that renders specified components and communicates with a ReviewData object to modify
        ReviewData object's annotation and history.

        """
        self.more_components = OrderedDict()
        
    def columns_to_string(self, df, columns):
        new_df = df.copy()
        for c in columns:
            new_df[c] = new_df[c].astype(str)

        return new_df
        
    def run(
        self,
        review_data: ReviewDataInterface,
        annot_app_display_types_dict: Dict = None,
        autofill_dict: Dict = None,
        review_data_table_df: pd.DataFrame = None,
        review_data_table_page_size: int = 10,
        collapsable=True,
        auto_export: bool = True,
        auto_export_path: Union[Path, str] = None, 
        attributes_to_export: List = ['annot_df', 'history_df'],
        mode='external',
        host='0.0.0.0',
        port=8050,
    ):

        """
        Run the app

        Parameters
        ----------
        review_data: ReviewDataInterface
            ReviewData object to review with the app

        mode: {'inline', 'external', 'tab'}
            How to display the dashboard

        host: str
            Host address

        port: int
            Port access number

        annot_app_display_types_dict: Dict
            at run time, determines how the inputs for annotations will be displayed

            Format: {annot_name: app_display_type}

            Rules:
                - annot_name: must be a key in ReviewData.data.annot_col_config_dict
                - app_display_type: see valid_annotation_app_display_types

        autofill_dict: Dict
            At run time, adds buttons for each component included and if clicked, defines how to autofill
            annotations for the review_data object using the specified data corresponding to the component
            
            Format: {button_name_1: {annot_name: State()},
                     button_name_2: {annot_name: State(), another_annot_name: 'a literal value'}}
            
            Rules:
                - button_name:  must be a name of a button
                - annot_name:      must be the name of an annotation type in the review_data (ReviewData) data
                                   annot_col_config_dict
                - autofill values: State()'s referring to objects in the component named component name, or a 
                                   valid literal value according to the DataAnnotation object's validation method.
                                   
        auto_export: bool, default=False
            Whether to auto export on save to path set by argument auto_export_path
            
        auto_export_path: Union[Path, str]
            Path to export data if argument auto_export=True
            
        attributes_to_export: List
            List of attributes from the data object to automatically export
        """
        multi_type_columns = [c for c in annot_app_display_types_dict.keys() if review_data.data.annot_col_config_dict[c].annot_value_type == 'multi']
        
        def get_history_display_table(subject_index_value):
            filtered_history_df = review_data.data.history_df.loc[
                review_data.data.history_df['index'] == subject_index_value
            ].loc[::-1]
            
            return self.columns_to_string(filtered_history_df, multi_type_columns)
        
        
        # app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        reviewed_data_df = pd.DataFrame(index=review_data.data.index, columns=['label'])
        reviewed_data_df['label'] = reviewed_data_df.apply(lambda r: self.gen_dropdown_labels(review_data, r), 
                                                           axis=1)
        reviewed_data_df.index.name = 'value'
        
        app.layout, annotation_panel_component, autofill_buttons, gen_autofill_states, autofill_literals = \
            self.gen_layout(
                review_data,
                reviewed_data_df,
                annot_app_display_types_dict,
                autofill_dict,
                review_data_table_df=review_data_table_df,
                review_data_table_page_size=review_data_table_page_size,
                collapsable=collapsable,
                multi_type_columns=multi_type_columns
            )
        
        app.title = review_data.data_pkl_fn.split('/')[-1].split('.')[0]
        
        if auto_export:
            if auto_export_path is None:
                auto_export_path = f"{review_data.data_pkl_fn.rsplit('.', 1)[0]}.auto_export"
                print(f'Setting auto_export_path to {auto_export_path}')
            if not os.path.exists(auto_export_path):
                print(f'Making directory {auto_export_path} for auto exporting.')
                os.mkdir(auto_export_path)
            
            print(f'Using {auto_export_path} for auto exporting.')
        
        def validate_callback_outputs(
            component_output, 
            component, 
            which_callback='callback function(s)'
        ):
            if not isinstance(component_output, list) or (len(component.callback_output) != len(component_output)):
                raise ValueError(
                    f'Component ({component.name}) {which_callback} does not return '
                     'the same length output as component\'s callback_output.\n'
                     'Make sure your output from the callback function is a list, '
                     'and the values correspond to your defined callback_outputs list\n'
                     f'Expected output: {component.callback_output}\n'
                     f'Runtime callback output: {component_output}\n'
                )
            return

        more_component_inputs = {
            c.name: c.callback_input + c.callback_state + c.callback_state_external for c_name, c in self.more_components.items()
        }

        more_component_inputs_as_states = {
            c_name: [
                State(**{'component_id': c.component_id, 'component_property': c.component_property}) for c in c_list
            ] for c_name, c_list in more_component_inputs.items()
        }

        more_component_outputs = {c.name: c.callback_output for c_name, c in self.more_components.items()}
               
        def update_components(output_dict, subject_index_value, more_component_inputs_as_states):
            output_dict['more_component_outputs'] = {
                c.name: list(np.full(len(c.callback_output), dash.no_update)) for c_name, c in self.more_components.items()
            }
            
            for c_name, component in self.more_components.items():
                if component.new_data_callback is not None:
                    component_output = component.new_data_callback(
                        review_data.data,
                        subject_index_value,
                        *more_component_inputs_as_states[component.name]
                    )
                    validate_callback_outputs(component_output, component, which_callback='new_data_callback')
                    output_dict['more_component_outputs'][component.name] = component_output
            
            history_df = get_history_display_table(subject_index_value)
            output_dict['history_table'] = history_df.to_dict('records')
            output_dict['history_table_selected_row_state'] = []
            
            if history_df.empty:
                output_dict['annot_panel'] = {
                    annot_name: annot_display_component.default_display_value if \
                    annot_display_component.default_display_value is not None else '' \
                    for annot_name, annot_display_component in annot_app_display_types_dict.items() 
                }
            else:
                current_annotations = review_data.data.annot_df.loc[subject_index_value, list(annot_app_display_types_dict.keys())].to_dict()
                for annot_name, v in current_annotations.items():
                    if isinstance(v, list):
                        continue
                    
                    if (
                        (pd.isna(v) or (v == '') or (v is None)) and 
                        (annot_app_display_types_dict[annot_name].default_display_value is not None)
                    ):
                        current_annotations[annot_name] = annot_app_display_types_dict[annot_name].default_display_value
                    
                output_dict['annot_panel'] = current_annotations
            return output_dict

        ####### Callbacks

        @app.callback(
            output=dict(
                more_component_outputs=more_component_outputs,
                history_table=Output('APP-history-table', 'data', allow_duplicate=True),
                history_table_selected_row_state=Output('APP-history-table', 'selected_rows', allow_duplicate=True),
                annot_panel=annotation_panel_component.callback_output,
                review_data_selected_value=Output('APP-review-data-table', 'selected_rows', allow_duplicate=True),
                review_data_page_current=Output('APP-review-data-table', 'page_current', allow_duplicate=True)
            ),
            inputs=dict(
                dropdown_value=Input('APP-dropdown-data-state', 'value'),
                review_data_table_state=State('APP-review-data-table', 'data'),
                more_component_inputs_as_states=more_component_inputs_as_states,
                review_data_table_derived_virtual_data_state=State('APP-review-data-table', 'derived_virtual_data')
            ),
            prevent_initial_call=True,
        )
        def update_sample_via_dropdown(
            dropdown_value,
            review_data_table_state,
            more_component_inputs_as_states,
            review_data_table_derived_virtual_data_state
        ):
            """
            Update the whole dashboard with the corresponding data of 
            the selected subject in the dropdown menu
            """
            output_dict = {
                'review_data_page_current': dash.no_update,
                'review_data_selected_value': dash.no_update,
            }
            
            tmp_review_data_table_df = pd.DataFrame.from_records(review_data_table_state)
            subject_index_value = dropdown_value

            if review_data_table_df is not None: # if the table is being used
                index_tmp_review_data_table_df = tmp_review_data_table_df.loc[
                    tmp_review_data_table_df['index'] == subject_index_value
                ]
                output_dict['review_data_selected_value'] = index_tmp_review_data_table_df.index

                review_data_table_derived_virtual_df = pd.DataFrame.from_records(
                    review_data_table_derived_virtual_data_state)
                index_relative_review_data_table_df = review_data_table_derived_virtual_df.loc[
                    review_data_table_derived_virtual_df['index'] == subject_index_value,
                ]
                if not index_relative_review_data_table_df.empty:
                    output_dict['review_data_page_current'] = floor(index_relative_review_data_table_df.index[0] / review_data_table_page_size)


            output_dict = update_components(output_dict, subject_index_value, more_component_inputs_as_states)
            return output_dict
            

        @app.callback(
            output=dict(
                dropdown_value=Output('APP-dropdown-data-state', 'value', allow_duplicate=True),
                more_component_outputs=more_component_outputs,
                history_table=Output('APP-history-table', 'data', allow_duplicate=True),
                history_table_selected_row_state=Output('APP-history-table', 'selected_rows', allow_duplicate=True),
                annot_panel=annotation_panel_component.callback_output,
            ),
            inputs=dict(
                review_data_selected_value=Input('APP-review-data-table', 'selected_rows'),
                review_data_table_state=State('APP-review-data-table', 'data'),
                more_component_inputs_as_states=more_component_inputs_as_states,
            ),
            prevent_initial_call=True,
        )
        def update_sample_via_review_table(
            review_data_selected_value,
            review_data_table_state,
            more_component_inputs_as_states
        ):
            """
            Update the whole dashboard with the corresponding data of the selected subject in the dash table
            """
            output_dict = {}
            
            tmp_review_data_table_df = pd.DataFrame.from_records(review_data_table_state)
            subject_index_value = tmp_review_data_table_df.loc[review_data_selected_value[0], 'index'] 
            output_dict['dropdown_value'] = subject_index_value

            output_dict = update_components(output_dict, subject_index_value, more_component_inputs_as_states)
            return output_dict
            

        @app.callback(
            output=dict(
                history_table=Output('APP-history-table', 'data', allow_duplicate=True),
                dropdown_list_options=Output('APP-dropdown-data-state', 'options', allow_duplicate=True),
                review_data_table_data=Output('APP-review-data-table', 'data', allow_duplicate=True)
            ),
            inputs=dict(
                submit_annot_button=Input('APP-submit-button-state', 'n_clicks'),
                annot_input_state=annotation_panel_component.callback_state,
                dropdown_value=State('APP-dropdown-data-state', 'value'),
                review_data_table_state=State('APP-review-data-table', 'data'),
            ),
            prevent_initial_call=True,
        )
        def submit_button_annotation(
            submit_annot_button,
            annot_input_state,
            dropdown_value,
            review_data_table_state
        ):
            """
            Save current annotations to the annot_df field, and update the dropdown menu timestamp and history table
            """
            output_dict = {'review_data_table_data': dash.no_update}
            for annot_name in annot_app_display_types_dict.keys():
                annot_type = review_data.data.annot_col_config_dict[annot_name]
                # Convert type
                if annot_app_display_types_dict[annot_name].display_output_format is not None:
                    annot_input_state[annot_name] = annot_app_display_types_dict[annot_name].display_output_format(
                        annot_input_state[annot_name]
                    )
                
                validate_annot_data(annot_type, annot_input_state[annot_name])
                    
                new_annot_input_state = dict(annot_input_state)
                review_data._update(dropdown_value, new_annot_input_state)
                
                if auto_export:
                    review_data.export_data(auto_export_path, attributes_to_export=attributes_to_export, verbose=False)

                output_dict['history_table'] = get_history_display_table(dropdown_value).to_dict('records')
                
                reviewed_data_df.loc[dropdown_value, 'label'] = self.gen_dropdown_labels(
                    review_data,
                    reviewed_data_df.loc[dropdown_value]
                )
                output_dict['dropdown_list_options'] = reviewed_data_df.reset_index().to_dict('records')

                if review_data_table_df is not None:
                    tmp_review_data_table_df = pd.DataFrame.from_records(review_data_table_state).set_index('index')
                    tmp_review_data_table_df.loc[
                        dropdown_value,
                        review_data.data.annot_df.columns
                    ] = review_data.data.annot_df.loc[dropdown_value].values
                    output_dict['review_data_table_data'] = self.columns_to_string(
                        tmp_review_data_table_df, 
                        multi_type_columns
                    ).reset_index().to_dict('records')

                return output_dict

               
        @app.callback(
            output=dict(
                annot_panel=annotation_panel_component.callback_output
            ),
            inputs=dict(
                autofill_buttons=[Input(b.id, 'n_clicks') for b in autofill_buttons],
                autofill_states=gen_autofill_states,
            ),
            prevent_initial_call=True,
        )
        def autofill_annot(
            autofill_buttons,
            autofill_states
        ):
            """
            Detect and fill annotation panel from autofill trigger
            """
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            else:
                prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
            output_dict = {
                'annot_panel': {
                    annot_col: dash.no_update for annot_col in annot_app_display_types_dict.keys()
                }
            }
            component_name = prop_id.split('APP-autofill-')[-1]
            for autofill_annot_col, value in autofill_states[prop_id].items():
                output_dict['annot_panel'][autofill_annot_col] = value
                
            for autofill_annot_col, value in autofill_literals[prop_id].items():
                output_dict['annot_panel'][autofill_annot_col] = value
            return output_dict


        @app.callback(
            output=dict(
                annot_panel=annotation_panel_component.callback_output
            ),
            inputs=dict(
                revert_annot_button=Input('APP-revert-annot-button', 'n_clicks'),
                history_table_selected_row_state=State('APP-history-table', 'selected_rows'),
                dropdown_value=State('APP-dropdown-data-state', 'value')
            ),
            prevent_initial_call=True,
        )
        def revert_annot(
            revert_annot_button,
            history_table_selected_row_state,
            dropdown_value
        ):
            """
            Revert the annotation fields to a previous annotation by selecting the row in the history table
            """
            output_dict = {
                'annot_panel': {
                    annot_col: dash.no_update for annot_col in annot_app_display_types_dict.keys()
                }
            }
            if len(history_table_selected_row_state) > 0:
                history_df = review_data.data.history_df.loc[review_data.data.history_df['index'] == dropdown_value].loc[::-1].reset_index()
                output_dict['annot_panel'] = history_df.iloc[
                    history_table_selected_row_state[0]
                ][review_data.data.annot_df.columns].fillna('').to_dict()
                
            return output_dict


        @app.callback(
            output=dict(
                annot_panel=annotation_panel_component.callback_output
            ),
            inputs=dict(
                clear_annot_button=Input('APP-clear-annot-button', 'n_clicks'),
            ),
            prevent_initial_call=True,
        )
        def clear_annot(clear_annot_button):
            """
            Clear the data currently entered in the annotation panel
            """
            output_dict = {}
            output_dict['annot_panel'] = {annot_col: '' for annot_col in review_data.data.annot_df.columns}
            return output_dict

               
        @app.callback(
            output=dict(
                more_component_outputs=more_component_outputs
            ),
            inputs=dict(
                more_component_inputs=more_component_inputs,
                dropdown_value=State('APP-dropdown-data-state', 'value'),
            ),
            prevent_initial_call=True,
        )
        def internal_update_components(more_component_inputs, dropdown_value):
            """
            Update triggered component
            """
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            else:
                prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

            output_dict = {
                'more_component_outputs': {
                    c.name: list(np.full(len(c.callback_output), dash.no_update)) for c_name, c in self.more_components.items()
                }
            }
            for c_name, component in self.more_components.items():
                if sum([ci.component_id == prop_id for ci in component.callback_input]) > 0:
                    if component.internal_callback is None:
                        raise ValueError(
                            f'Component ({component.name}) has Inputs that change ({prop_id}), '
                            f'but no internal_callback defined to handle it.'
                            f'Either remove Input "{prop_id}" from "{component.name}.callback_input" attribute, '
                            f'or define a callback function'
                        )

                    component_output = component.internal_callback(
                        review_data.data,
                        dropdown_value,
                        *more_component_inputs[component.name]
                    )

                    validate_callback_outputs(component_output, component, which_callback='internal_callback')
                    output_dict['more_component_outputs'][component.name] = component_output
            return output_dict

        jupyter_dash.default_mode = mode
        app.run(host=host, port=port, debug=True)
        
    def gen_layout(
        self,
        review_data: ReviewDataInterface,
        reviewed_data_df: pd.DataFrame,
        annot_app_display_types_dict: Dict,
        autofill_dict: Dict,
        review_data_table_df: pd.DataFrame=None,
        review_data_table_page_size: int = 10,
        collapsable=True,
        multi_type_columns=[]
    ):
        """
        Generate layout of the dashboard
        """
        review_data_title = html.Div([html.H1(review_data.data_pkl_fn.split('/')[-1].split('.')[0])])
        review_data_path = html.Div([html.P(f'Path: {review_data.data_pkl_fn}')])
        review_data_description = html.Div([html.P(f'Description: {review_data.data.description}')])
        dropdown = html.Div(dcc.Dropdown(options=reviewed_data_df.reset_index().to_dict('records'),
                                         value=None, 
                                         id='APP-dropdown-data-state'))
        
        dropdown_component = AppComponent(name='APP-dropdown-component',
                                          layout=[dropdown], 
                                          use_name_as_title=False)

        if review_data_table_df is not None:
            if set(review_data_table_df.index.tolist()) != set(reviewed_data_df.index.tolist()):
                raise ValueError(
                    f'{review_data_table_df.index} has missing or extra index values. '
                    f'Index values should look like: {reviewed_data_df.index.tolist()}'
                )
            new_review_data_table_df = review_data_table_df.copy()
            new_review_data_table_df.index.name = 'index'
            new_review_data_table_df = pd.concat(
                [
                    new_review_data_table_df, 
                    self.columns_to_string(review_data.data.annot_df, multi_type_columns)
                ], 
                axis=1
            )
            review_data_table_data = new_review_data_table_df.reset_index().to_dict('records')
            review_data_table_columns = new_review_data_table_df.reset_index().columns.tolist()
            style = {'display': 'block'}

        else:
            review_data_table_layout = html.Div(html.H1('None'), style={'display': 'none'})
            review_data_table_data = []
            review_data_table_columns = []
            style = {'display': 'none'}

        review_data_table_layout = html.Div(
            dash.dash_table.DataTable(
                id='APP-review-data-table',
                data=review_data_table_data,
                columns=[
                    {"name": i, "id": i, "deletable": False, "selectable": False} for i in review_data_table_columns
                ],
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="single",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=review_data_table_page_size,
                style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                },
            ),
            style=style,
        )

        review_data_table_component = AppComponent(
            name='APP-review-data-table-app-component',
            layout=review_data_table_layout,
            use_name_as_title=False,
        )


        history_table = html.Div([
            html.H2('History Table'),
            html.Button(
                'Revert to selected annotation',
                id=f'APP-revert-annot-button',
                n_clicks=0,
                style={"marginBottom": "15px"}),
            html.Button(
                'Clear annotation inputs',
                id=f'APP-clear-annot-button',
                n_clicks=0,
                style={"marginBottom": "15px"}),
            html.Div(
                dash.dash_table.DataTable(
                    id='APP-history-table',
                    data=pd.DataFrame().to_dict('records'),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": False} for i in review_data.data.history_df.columns
                    ],
                    editable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    row_selectable="single",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=5,
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    }
                ),
                style={"maxHeight": "400px", "overflow": "scroll"},
            )
        ])
        
        history_component = AppComponent(name='APP-history-component',
                                         layout=[history_table], 
                                         use_name_as_title=False)
        
        autofill_buttons, autofill_states, autofill_literals = self.gen_autofill_buttons_and_states(review_data,
                                                                                                    autofill_dict)

        annotation_panel_component = self.gen_annotation_panel_component(review_data, autofill_buttons,
                                                                         annot_app_display_types_dict)
        
        
        if collapsable:
            more_components_layout = dbc.Accordion(
                [dbc.AccordionItem(c.layout, title=c_name) for c_name, c in self.more_components.items()], 
                always_open=True, 
                start_collapsed=False,
                active_item=[f'item-{i}' for i in range(len(self.more_components.keys()))],
                style={'.accordion-button': {'font-size': 'xx-large'}}
            )
        else:
            more_components_layout = [dbc.Row(c.layout, style={"marginBottom": "15px"}) for c_name, c in self.more_components.items()]

        
        layout = html.Div(
            [
                dbc.Row([review_data_title], style={"marginBottom": "15px"}),
                dbc.Row([review_data_path], style={"marginBottom": "15px"}),
                dbc.Row([review_data_description], style={"marginBottom": "15px"}),
                dbc.Row([dropdown_component.layout], style={"marginBottom": "15px"}),
                dbc.Row([review_data_table_component.layout], style={"marginBottom": "15px"}),
                dbc.Row([dbc.Col(annotation_panel_component.layout, width=5),
                        dbc.Col(html.Div(history_component.layout), width=7)], 
                       style={"marginBottom": "15px"}),
                dbc.Row(more_components_layout)
            ],
             style={'marginBottom': 50, 'marginTop': 25, 'marginRight': 25, 'marginLeft': 25})
        
        
        all_ids = np.array(get_component_ids(layout))
        check_duplicates(all_ids, 'full component')

        return layout, annotation_panel_component, autofill_buttons, autofill_states, autofill_literals

    def gen_dropdown_labels(self, review_data: ReviewDataInterface, r: pd.Series):
        data_history_df = review_data.data.history_df[review_data.data.history_df["index"] == r.name]
        if not data_history_df.empty:
            return str(r.name) + f' (Last update: {data_history_df["timestamp"].tolist()[-1]})'
        else:
            return r.name
       
    def gen_autofill_buttons_and_states(self, review_data, autofill_dict):
        review_data_annot_names = list(review_data.data.annot_col_config_dict.keys())
            
        autofill_buttons = []
        autofill_states = {}
        autofill_literals = {}
        for autofill_button_name, button_autofill_dict in autofill_dict.items():

            # check keys in component_autofill_dict are in ReviewData annot columns
            annot_keys = np.array(list(button_autofill_dict.keys()))
            missing_annot_refs = np.array([annot_k not in review_data_annot_names for annot_k in annot_keys])
            if missing_annot_refs.any():
                raise ValueError(
                    f'Reference to annotation columns {annot_keys[np.argwhere(missing_annot_refs).flatten()]}'
                    f' do not exist in the Review Data Object.'
                    f'Available annotation columns are {review_data_annot_names}')

            # check states exist in layout
            button_autofill_states_dict = {annot_type: state for annot_type, state in button_autofill_dict.items()
                                              if isinstance(state, State)}
            button_autofill_state_ids = np.array(
                get_callback_io_ids(list(button_autofill_states_dict.values()), expected_io_type=State))

            all_ids = get_component_ids([c.layout for c_name, c in self.more_components.items()])
            missing_autofill_state_ids = [i for i in button_autofill_state_ids if i not in all_ids]
            if len(missing_autofill_state_ids) > 0:
                raise ValueError(f'State ids do not exist anywhere in the layout: '
                                 f'{missing_autofill_state_ids}\n')

            # check literal fill values are valid
            button_autofill_non_states_dict = {annot_type: item for annot_type, item in
                                               button_autofill_dict.items() if not isinstance(item, State)}

            for annot_type, non_state_value in button_autofill_non_states_dict.items():
                if isinstance(non_state_value, Input) or isinstance(non_state_value, Output):
                    raise ValueError(
                        f'Invalid autofill object {non_state_value} for annotation column "{annot_type}". '
                        'Either pass in a State() object or literal value')

                try:
                    review_data.data.annot_col_config_dict[annot_type].validate(non_state_value)
                except ValueError:
                    raise ValueError(
                        f'Autofill value for annotation column '
                        f'"{annot_type}" failed validation. '
                        f'Check validation for annotation column "{annot_type}": \n'
                        f'{review_data.data.annot_col_config_dict[annot_type]}')

            # Make button    
            autofill_button_component = html.Button(autofill_button_name,
                                                    id=f'APP-autofill-{autofill_button_name}',
                                                    n_clicks=0,
                                                    style={"marginBottom": "15px"})
            autofill_buttons += [autofill_button_component]
            autofill_states[autofill_button_component.id] = button_autofill_states_dict
            autofill_literals[autofill_button_component.id] = button_autofill_non_states_dict

        if len(autofill_buttons) == 0:
            autofill_button_component = html.Button(
                'dummy button - required for autofill callback',
                id=f'APP-autofill-dummy-button',
                disabled=True, hidden=True,
                style={"marginBottom": "15px"}
            )
            autofill_buttons += [autofill_button_component]
            
        return autofill_buttons, autofill_states, autofill_literals


    def gen_annotation_panel_component(
        self,
        review_data: ReviewDataInterface,
        autofill_buttons: [html.Button],
        annot_app_display_types_dict: Dict
    ):

        if len(annot_app_display_types_dict) == 0:
            raise ValueError(
                f'annot_app_display_types_dict is empty. '
                f'Make sure to run reviewer.set_default_review_data_annotations_configuration(), '
                f'or manually set up your annotations'
            )
        submit_annot_button = html.Button(id='APP-submit-button-state', 
                                          n_clicks=0, 
                                          children='Submit', 
                                          style={"marginBottom": "15px"})
        
        def annotation_display_component_input(annot_name, annot, annot_app_display_type):
            input_component_id = f"APP-{annot_name}-{annot_app_display_type}-input-state"
            
            input_component = annot_app_display_type.gen_input_component(
                annot, 
                component_id=input_component_id, 
            )
                
            return dbc.Row([dbc.Label(annot_name, html_for=input_component_id, width=2),
                            dbc.Col(input_component)],
                           style={"margin-bottom": "15px"})
        
        panel_components = autofill_buttons + [
            annotation_display_component_input(
                annot_name, review_data.data.annot_col_config_dict[annot_name], display_type
            ) for annot_name, display_type in annot_app_display_types_dict.items()
        ] + [submit_annot_button]
        
        panel_inputs = [Input('APP-submit-button-state', 'nclicks')]
        return AppComponent(
            name='APP-Panel',
            layout=panel_components,
            callback_output={
                annot_name: Output(f"APP-{annot_name}-{display_type}-input-state", "value", allow_duplicate=True)
                for annot_name, display_type in annot_app_display_types_dict.items()},
            callback_input=panel_inputs,
            callback_state={
                annot_name: State(f"APP-{annot_name}-{display_type}-input-state", "value") for
                    annot_name, display_type in annot_app_display_types_dict.items()},
            use_name_as_title=False
        )
    
    
    def add_component(
        self, 
        component: AppComponent, 
        **kwargs
    ):
        """
        component: An AppComponent object to include in the app
        **kwargs: include more arguments for the component's callback functions 
        """
        
        all_component_names = [c_name for c_name, c in self.more_components.items()]
        
        if component.name in all_component_names:
            warnings.warn(f'There is a component named "{component.name}" already in the app. '
                          f'Change the name of the component to add it to the current layout')
            return
        
        ids_with_reserved_prefix_list = np.array(['APP-' in i for i in component.all_component_ids])
        if ids_with_reserved_prefix_list.any():
            raise ValueError(f'Some ids use reserved keyword "APP-". Please change the id name\n'
                             f'Invalid component ids: '
                             f'{component.all_component_ids[np.argwhere(ids_with_reserved_prefix_list).flatten()]}')
        
        new_component = copy.deepcopy(component)
        if component.new_data_callback is not None:
            new_component.new_data_callback = lambda *args: component.new_data_callback(*args, **kwargs)
        else:
            new_component.new_data_callback = None
            
        if component.internal_callback is not None:
            new_component.internal_callback = lambda *args: component.internal_callback(*args, **kwargs)
        else:
            new_component.internal_callback = None

        self.more_components[component.name] = new_component
        all_ids = get_component_ids([c.layout for c_name, c in self.more_components.items()])
        check_duplicates(all_ids, f'ids found in previously added component from component named {component.name}')
        
    def add_table_from_path(self, data_table_source, table_title, component_id, table_fn_col, table_cols):
        """
        Parameters
        ----------
        data_table_source: attribute name of the dataframe to use from the Data object
        table_title:     Title of the table
        component_id: component name for the table
        table_fn_col:   column in review_data data dataframe with file path with table to display
        table_cols:     columns to display in table from table_fn_col
        """
        table = html.Div(dbc.Table.from_dataframe(pd.DataFrame()),
                         id=component_id)
        self.add_component(
            AppComponent(
                table_title,
                table,
                new_data_callback=lambda data, idx: [dbc.Table.from_dataframe(
                    pd.read_csv(getattr(data, data_table_source).loc[idx, table_fn_col],
                                sep='\t',
                                encoding='iso-8859-1')[table_cols])],
                callback_output=[Output(component_id, 'children')]
            )
        )


def get_component_ids(component: Union[List, Tuple]):
    if isinstance(component, list) or isinstance(component, tuple):
        id_list = []
        for comp in component:
            id_list += get_component_ids(comp)
        return id_list
    elif isinstance(component, str) or isinstance(component, int) or isinstance(component, float):
        return []
    elif 'children' in component.__dict__.keys(): 
        children_ids = get_component_ids(component.children)
        if 'id' in component.__dict__.keys():
            children_ids += [component.id]
        return children_ids
    else:
        return [component.id] if 'id' in component.__dict__.keys() else []


def get_callback_io_ids(io_list, expected_io_type):
    
    if isinstance(io_list, list):
        items = io_list
    elif isinstance(io_list, dict):
        items = [item for k, item in io_list.items()]
    else:
        raise ValueError(f'Input list is not a list or dictionary: {io_list}')

    wrong_io_type = np.array([not isinstance(item, expected_io_type) for item in items])
    if wrong_io_type.any():
        raise ValueError(f'Some or all items in list are not the expected type {expected_io_type}: {items}')
        
    return [item.component_id for item in items]


def check_callback_io_id_in_list(ids, all_ids, ids_type: str, all_ids_type: str):
    ids_not_in_all_ids = np.array(ids)[np.argwhere(np.array([id_name not in all_ids for id_name in ids])).flatten()]
    if len(ids_not_in_all_ids) > 0:
        raise ValueError(f"{ids_type} ids not in {all_ids_type}:\n"
                         f"{ids_type} ids: {ids_not_in_all_ids}\n"
                         f"{all_ids_type} ids: {all_ids}")


def check_duplicates(a_list, list_type: str):
    values, counts = np.unique(a_list, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"Duplicate {list_type}: {values[np.argwhere(counts > 1)].flatten()}")


def check_duplicate_objects(a_list, list_type: str):
    check_duplicates([c.__str__() for c in a_list], list_type=list_type)
