import pandas as pd
import numpy as np
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
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

from .ReviewDataInterface import ReviewDataInterface

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
        
        check_duplicate_objects(callback_output, 'callback_output_objects')
        callback_output_ids = get_callback_io_ids(callback_output, expected_io_type=Output)
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
        self.callback_output = callback_output
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
            
    def run(self,
            review_data: ReviewDataInterface,
            annot_app_display_types_dict: Dict = None,
            autofill_dict: Dict = None,
            review_data_table_df: pd.DataFrame = None,
            review_data_table_page_size: int = 10,
            mode='external',
            host='0.0.0.0',
            port=8050):

        """
        Run the app

        Parameters
        ----------
        review_data: ReviewDataInterface
            ReviewData object to review with the app

        mode: {'inline', 'external'}
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

        """
        
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
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
            )
        app.title = review_data.data_pkl_fn.split('/')[-1].split('.')[0]
        
        def validate_callback_outputs(component_output, 
                                      component, 
                                      which_callback='callback function(s)'):
            if not isinstance(component_output, list) or (len(component.callback_output) != len(component_output)):
                raise ValueError(f'Component ({component.name}) {which_callback} does not return '
                                 'the same length output as component\'s callback_output.\n'
                                 'Make sure your output from the callback function is a list, '
                                 'and the values correspond to your defined callback_outputs list\n'
                                 f'Expected output: {component.callback_output}\n'
                                 f'Runtime callback output: {component_output}\n')
            return
        
        @app.callback(output=dict(history_table=Output(f'APP-history-table', 'children'),
                                  annot_panel=annotation_panel_component.callback_output,
                                  dropdown_list_options=Output(f'APP-dropdown-data-state', 'options'),
                                  dropdown_value=Output('APP-dropdown-data-state', 'value'),
                                  review_data_selected_value=Output('APP-review-data-table', 'selected_rows'),
                                  review_data_page_current=Output('APP-review-data-table', 'page_current'),
                                  more_component_outputs={c.name: c.callback_output for c_name, c in
                                                          self.more_components.items()}),
                      inputs=dict(dropdown_value=Input('APP-dropdown-data-state', 'value'),
                                  review_data_selected_value=Input('APP-review-data-table', 'selected_rows'),
                                  review_data_table_state=Input('APP-review-data-table', 'data'),
                                  autofill_buttons=[Input(b.id, 'n_clicks') for b in autofill_buttons],
                                  autofill_states=gen_autofill_states,
                                  submit_annot_button=Input('APP-submit-button-state', 'n_clicks'),
                                  annot_input_state=annotation_panel_component.callback_state,
                                  more_component_inputs={
                                      c.name: c.callback_input + c.callback_state + c.callback_state_external for
                                      c_name, c in
                                      self.more_components.items()}))
        def component_callback(dropdown_value,
                               review_data_selected_value,
                               review_data_table_state,
                               autofill_buttons,
                               autofill_states,
                               submit_annot_button, 
                               annot_input_state, 
                               more_component_inputs):
            
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            else:
                prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            output_dict = {'history_table': dash.no_update,
                           'annot_panel': {annot_col: dash.no_update for annot_col in
                                           review_data.data.annot_df.columns},
                           'dropdown_list_options': dash.no_update,
                           'dropdown_value': dash.no_update,
                           'review_data_selected_value': dash.no_update,
                           'review_data_page_current': dash.no_update,
                           'more_component_outputs': {c.name: list(np.full(len(c.callback_output), dash.no_update)) for
                                                      c_name, c in self.more_components.items()},
                           }

            if (prop_id == 'APP-dropdown-data-state') or (prop_id == 'APP-review-data-table'):
                review_data_table_df = pd.DataFrame.from_records(review_data_table_state)
                if prop_id == 'APP-dropdown-data-state':
                    subject_index_value = dropdown_value

                    if 'index' in review_data_table_df.columns:
                        output_dict['review_data_selected_value'] = review_data_table_df.loc[
                            review_data_table_df['index'] == subject_index_value,
                            ].index
                        output_dict['review_data_page_current'] = floor(output_dict['review_data_selected_value'][0] / review_data_table_page_size)
                else:
                    subject_index_value = review_data_table_df.loc[review_data_selected_value[0], 'index'].item()
                    output_dict['dropdown_value'] = subject_index_value


                for c_name, component in self.more_components.items():
                    if component.new_data_callback is not None:
                        component_output = component.new_data_callback(review_data.data,
                                                                       subject_index_value,
                                                                       *more_component_inputs[component.name])
                        validate_callback_outputs(component_output, component, which_callback='new_data_callback')
                        output_dict['more_component_outputs'][component.name] = component_output

                output_dict['history_table'] = dbc.Table.from_dataframe(
                    review_data.data.history_df.loc[review_data.data.history_df['index'] == subject_index_value].loc[::-1])
                output_dict['annot_panel'] = {annot_col: '' for annot_col in review_data.data.annot_df.columns}
                            
            elif (prop_id == 'APP-submit-button-state') & (submit_annot_button > 0):
                for annot_name in annot_app_display_types_dict.keys():
                    annot_type = review_data.data.annot_col_config_dict[annot_name]
                    annot_type.validate(annot_input_state[annot_name])
                review_data._update(dropdown_value, annot_input_state)
                output_dict['history_table'] = dbc.Table.from_dataframe(
                    review_data.data.history_df.loc[review_data.data.history_df['index'] == dropdown_value].loc[::-1])
                reviewed_data_df.loc[dropdown_value, 'label'] = self.gen_dropdown_labels(review_data,
                                                                                         reviewed_data_df.loc[
                                                                                             dropdown_value])
                output_dict['dropdown_list_options'] = reviewed_data_df.reset_index().to_dict('records')
            elif 'APP-autofill-' in prop_id:
                component_name = prop_id.split('APP-autofill-')[-1]
                for autofill_annot_col, value in autofill_states[prop_id].items():
                    output_dict['annot_panel'][autofill_annot_col] = value
                    
                for autofill_annot_col, value in autofill_literals[prop_id].items():
                    output_dict['annot_panel'][autofill_annot_col] = value
            else:
                for c_name, component in self.more_components.items():
                    if sum([ci.component_id == prop_id for ci in component.callback_input]) > 0:
                        if component.internal_callback is None:
                            raise ValueError(
                                f'Component ({component.name}) has Inputs that change ({prop_id}), '
                                f'but no internal_callback defined to handle it.'
                                f'Either remove Input "{prop_id}" from "{component.name}.callback_input" attribute, '
                                f'or define a callback function')

                        component_output = component.internal_callback(review_data.data,
                                                                       dropdown_value,
                                                                       *more_component_inputs[component.name])

                        validate_callback_outputs(component_output, component, which_callback='internal_callback')
                        output_dict['more_component_outputs'][component.name] = component_output
                pass
            return output_dict
        
        app.run_server(mode=mode, host=host, port=port, debug=True) 
        
    def gen_layout(self,
                   review_data: ReviewDataInterface,
                   reviewed_data_df: pd.DataFrame,
                   annot_app_display_types_dict: Dict,
                   autofill_dict: Dict,
                   review_data_table_df: pd.DataFrame=None,
                   review_data_table_page_size: int = 10
                   ):
        
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
            review_data_table_data = review_data_table_df.reset_index().to_dict('records')
            review_data_table_columns = review_data_table_df.reset_index().columns.tolist()
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
            ),
            style=style,
        )

        review_data_table_component = AppComponent(
            name='APP-review-data-table-app-component',
            layout=review_data_table_layout,
            use_name_as_title=False,
        )


        history_table = html.Div([html.H2('History Table'),
                                  html.Div([dbc.Table.from_dataframe(
                                      pd.DataFrame(columns=review_data.data.history_df.columns))],
                                           style={"maxHeight": "400px", "overflow": "scroll"},
                                           id='APP-history-table')
                                  ])
        
        history_component = AppComponent(name='APP-history-component',
                                         layout=[history_table], 
                                         use_name_as_title=False)
        
        autofill_buttons, autofill_states, autofill_literals = self.gen_autofill_buttons_and_states(review_data,
                                                                                                    autofill_dict)

        annotation_panel_component = self.gen_annotation_panel_component(review_data, autofill_buttons,
                                                                         annot_app_display_types_dict)
        
        # TODO: save current view as html
        layout = html.Div([dbc.Row([review_data_title], style={"marginBottom": "15px"}),
                           dbc.Row([review_data_path], style={"marginBottom": "15px"}),
                           dbc.Row([review_data_description], style={"marginBottom": "15px"}),
                           dbc.Row([dropdown_component.layout], style={"marginBottom": "15px"}),
                           dbc.Row([review_data_table_component.layout], style={"marginBottom": "15px"}),
                           dbc.Row([dbc.Col(annotation_panel_component.layout, width=5),
                                    dbc.Col(html.Div(history_component.layout), width=7)], 
                                   style={"marginBottom": "15px"}),
                           dbc.Row([dbc.Row(c.layout,
                                            style={"marginBottom": "15px"}) for c_name, c in
                                    self.more_components.items()])],
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
                    f' do not existin in the Review Data Object.'
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
            
        return autofill_buttons, autofill_states, autofill_literals


    def gen_annotation_panel_component(self,
                                       review_data: ReviewDataInterface,
                                       autofill_buttons: [html.Button],
                                       annot_app_display_types_dict: Dict
                                       ):

        submit_annot_button = html.Button(id='APP-submit-button-state', 
                                          n_clicks=0, 
                                          children='Submit', 
                                          style={"marginBottom": "15px"})
        
        def annotation_input(annot_name, annot, annot_app_display_type):
            
            input_component_id = f"APP-{annot_name}-{annot_app_display_type}-input-state"
            
            if annot_app_display_type == 'textarea':
                input_component = dbc.Textarea(size="lg", 
                                               id=input_component_id,
                                               value=annot.default),
            elif annot_app_display_type == 'text':
                input_component = dbc.Input(type="text",
                                            id=input_component_id,
                                            placeholder=f"Enter {annot_name}",
                                            value=annot.default,
                                            )
            elif annot_app_display_type == 'number':
                input_component = dbc.Input(type="number", 
                                            id=input_component_id, 
                                            placeholder=f"Enter {annot_name}",
                                            value=annot.default)
            elif annot_app_display_type == 'checklist':
                default = [] if annot.default is None else annot.default
                input_component = dbc.Checklist(options=[{"label": f, "value": f} for f in annot.options],
                                                id=input_component_id, 
                                                value=default),
            elif annot_app_display_type == 'radioitem':
                input_component = dbc.RadioItems(
                                                options=[{"label": f, "value": f} for f in annot.options],
                                                value=annot.default,
                                                id=input_component_id,
                                            )
            elif annot_app_display_type == 'select':
                input_component = dbc.Select(
                                            options=[{"label": f, "value": f} for f in annot.options],
                                            value=annot.default,
                                            id=input_component_id,
                                            ),
            else:
                raise ValueError(f'Invalid annotation type "{annot.annot_type}"')
                
            return dbc.Row([dbc.Label(annot_name, html_for=input_component_id, width=2),
                            dbc.Col(input_component)],
                           style={"margin-bottom": "15px"})
        
        panel_components = autofill_buttons + [
            annotation_input(
                annot_name, review_data.data.annot_col_config_dict[annot_name], display_type
            ) for annot_name, display_type in annot_app_display_types_dict.items()
        ] + [submit_annot_button]
        
        panel_inputs = [Input('APP-submit-button-state', 'nclicks')]
        return AppComponent(
            name='APP-Panel',
            layout=panel_components,
            callback_output={
                annot_name: Output(f"APP-{annot_name}-{display_type}-input-state", "value")
                for annot_name, display_type in annot_app_display_types_dict.items()},
            callback_input=panel_inputs,
            callback_state={
                annot_name: State(f"APP-{annot_name}-{display_type}-input-state", "value") for
                    annot_name, display_type in annot_app_display_types_dict.items()},
            use_name_as_title=False
        )
        
    def add_component(self, 
                      component: AppComponent, 
                      **kwargs):
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
        self.add_component(AppComponent(table_title,
                                        table,
                                        new_data_callback=lambda data, idx: [dbc.Table.from_dataframe(
                                            pd.read_csv(getattr(data, data_table_source).loc[idx, table_fn_col],
                                                        sep='\t',
                                                        encoding='iso-8859-1')[table_cols])],
                                        callback_output=[Output(component_id, 'children')]
                                        ))


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
