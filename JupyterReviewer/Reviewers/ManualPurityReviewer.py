from JupyterReviewer.Data import Data, DataAnnotation
from JupyterReviewer.ReviewDataApp import ReviewDataApp, AppComponent
from JupyterReviewer.DataTypes.GenericData import GenericData

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from JupyterReviewer.ReviewerTemplate import ReviewerTemplate
from JupyterReviewer.lib.plot_cnp import plot_acr_interactive

from rpy2.robjects import r, pandas2ri
import os
import pickle
from typing import List, Dict


csize = {'1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276, '5': 180915260,
        '6': 171115067, '7': 159138663, '8': 146364022, '9': 141213431, '10': 135534747,
        '11': 135006516, '12': 133851895, '13': 115169878, '14': 107349540, '15': 102531392,
        '16': 90354753, '17': 81195210, '18': 78077248, '19': 59128983, '20': 63025520,
        '21': 48129895, '22': 51304566, '23': 156040895, '24': 57227415}


def gen_data_summary_table(data: GenericData,
                           data_id, 
                           cols):
    data_df = data.df
    r = data_df.loc[data_id]
    sample_data_df = r[cols].to_frame()
    sample_data_df[r.name] = sample_data_df[r.name].astype(str)
    sample_data_df['Console_link'] = ''
    for attr, value in sample_data_df.iterrows():
        if 'gs://' in value[r.name]:
            path = value[r.name].split('/', 2)[-1]
            sample_data_df.loc[attr, 'Console_link'] = f"https://console.cloud.google.com/storage/browser/_details/{path}"
    sample_data_df['Console_link'] = sample_data_df['Console_link'].apply(lambda url: html.A(html.P(url),
                                                                              href=url,
                                                                              target="_blank"))
    return [[html.H1(f'{r.name} Data Summary'), dbc.Table.from_dataframe(sample_data_df.reset_index())]]


def plot_cnp_histogram(fig, row, col, 
                       seg_df, 
                       mu_major_col, 
                       mu_minor_col, 
                       length_col,
                       max_mu=2, step=0.05):
    # bin over 
    mu_bins = np.arange(0, max_mu + step, step)
    mu_bins_counts = {b: 0 for b in mu_bins}
    for _, r in seg_df.iterrows():
        mu_bins_counts[mu_bins[np.argmax(r[mu_major_col] < mu_bins) - 1]] += r[length_col]
        mu_bins_counts[mu_bins[np.argmax(r[mu_minor_col] < mu_bins) - 1]] += r[length_col]
    
    # add half step to make them centered around the bin
    half_step = step / 2.0
    mu_bins_counts = {mu_bin + half_step: val for mu_bin, val in mu_bins_counts.items()}
    
    mu_bin_counts_df = pd.DataFrame.from_dict(mu_bins_counts, 
                                              orient='index',
                                              columns=['count'])
    mu_bin_counts_df.index.name = 'mu_bin'
    mu_bin_counts_df = mu_bin_counts_df.reset_index()
    
    bar_trace = go.Bar(x=mu_bin_counts_df['count'], y=mu_bin_counts_df['mu_bin'], orientation='h')
    fig.add_trace(bar_trace, row=row, col=col)
    fig.update_xaxes(title_text='Length Count',  
                     row=row, col=col)
    fig.update_layout(showlegend=False)


def gen_cnp_figure(acs_fn,
                   sigmas=True, 
                   mu_major_col='mu.major', 
                   mu_minor_col='mu.minor', 
                   length_col='length',
#                    csize=csize
                  ):
    
    seg_df = pd.read_csv(acs_fn, sep='\t')
    layout = go.Layout(
            plot_bgcolor='rgba(0,0,0,0)',
        )
    cnp_fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.77, 0.25])
    plot_acr_interactive(seg_df, cnp_fig, csize, sigmas=sigmas, row=0, col=0)
    
    plot_cnp_histogram(cnp_fig, 1, 2,
                       seg_df,
                       mu_major_col, 
                       mu_minor_col, 
                       length_col)
    
    return cnp_fig

def gen_absolute_component(data_df, 
                           data_id, 
                           selected_row_array, # dash app parameters come first
                           rdata_tsv_fn,
                           cnp_fig_pkl_fn_col,
                           mut_fig_pkl_fn_col
                          ):
    r = data_df.loc[data_id]
    absolute_rdata_df = pd.read_csv(r[rdata_tsv_fn], sep='\t')

    cnp_fig = load_pickle(r[cnp_fig_pkl_fn_col])

    mut_fig = load_pickle(r[mut_fig_pkl_fn_col])


    # add 1 and 0 lines
    mut_fig_with_lines = go.Figure(mut_fig)
    cnp_fig_with_lines = go.Figure(cnp_fig)
    solution_data = absolute_rdata_df.iloc[selected_row_array[0]]
    i = 0
    line_height = solution_data['0_line']
    while line_height < 2:
        line_height = solution_data['0_line'] + (solution_data['step_size'] * i)
        cnp_fig_with_lines.add_hline(y=line_height, 
                                     line_dash="dash", 
                                     line_color='black',
                                     line_width=1
                                    )
        i += 1
        
    half_1_line = solution_data['alpha'] / 2.0
    mut_fig_with_lines.add_hline(y=half_1_line,
                                line_dash="dash",
                                line_color='black',
                                line_width=1)

    mut_fig_with_lines.update_yaxes(range=[0, half_1_line * 2])

    return [absolute_rdata_df.to_dict('records'), 
            cnp_fig_with_lines,
            mut_fig_with_lines,
            solution_data['alpha'],
            solution_data['tau_hat'], 
            [0]]

def internal_gen_absolute_component(data: GenericData,
                                   data_id,
                                   selected_row_array, # dash app parameters come first
                                   rdata_tsv_fn,
                                   cnp_fig_pkl_fn_col,
                                   mut_fig_pkl_fn_col
                                  ):
    output_data = gen_absolute_component(
        data,
        data_id,
        selected_row_array,  # dash app parameters come first
        rdata_tsv_fn,
        cnp_fig_pkl_fn_col,
        mut_fig_pkl_fn_col)
    output_data[-1] = selected_row_array
    return output_data


def load_pickle(fn):
    return pickle.load(open(fn, "rb"))


def validate_purity(x):
    return (x >= 0) and (x <= 1)


def validate_ploidy(x):
    return x >=0
    

class ManualPurityReviewer(ReviewerTemplate):

    def gen_data(self,
                 description: str,
                 df: pd.DataFrame,
                 index: List,
                 preprocess_data_dir: str,
                 acs_col,
                 maf_col,
                 rdata_fn_col,
                 reload_cnp_figs=False,
                 reload_mut_figs=False,
                 mut_fig_hover_data=[],
                 annot_df: pd.DataFrame = None,
                 annot_col_config_dict: Dict = None,
                 history_df: pd.DataFrame = None) -> GenericData:
        """
        Parameters
        ==========
        preprocess_data_dir: str or path
            Path to directory to store premade plots
        acs_col: str
            Column name in df with path to seg file produced by alleliccapseg
        maf_col: str
            Column name in df with path to mutation validator validated maf
        rdata_fn_col: str
            Column name in df with path to rdata produced by Absolute
        reload_mut_figs: bool
            Whether to regenerate the mutation figures again
        reload_cnp_figs: bool
            Whether to regenerate the copy number plot
        mut_fig_hover_data: List
            List of column names in the maf file (from maf_col in df) to display on hover in
            mutation figures

        Returns
        =======
        GenericData
            A data object
        """
        pandas2ri.activate()
        if not os.path.exists(preprocess_data_dir):
            os.mkdir(preprocess_data_dir)

        # 2. Process cnp figures
        cnp_figs_dir = f'{preprocess_data_dir}/cnp_figs'
        if not os.path.exists(cnp_figs_dir):
            os.mkdir(cnp_figs_dir)
            reload_cnp_figs = True
        else:
            print(f'cnp figs directory already exists: {cnp_figs_dir}')

        if reload_cnp_figs:
            print('Reloading cnp figs')
            for i, r in df.iterrows():
                output_fn = f'{cnp_figs_dir}/{i}.cnp_fig.pkl'
                fig = gen_cnp_figure(df.loc[i, acs_col])
                pickle.dump(fig, open(output_fn, "wb"))
                df.loc[i, f'cnp_figs_pkl'] = output_fn

        mut_figs_dir = f'{preprocess_data_dir}/mut_figs'
        if not os.path.exists(mut_figs_dir):
            os.mkdir(mut_figs_dir)
            reload_mut_figs = True
        else:
            print(f'mut figs directory already exists: {mut_figs_dir}')


        return GenericData(index=index,
                           description=description,
                           df=df,
                           annot_df=annot_df,
                           annot_col_config_dict=annot_col_config_dict,
                           history_df=history_df)

    def gen_review_app(self,
                       sample_info_cols,
                       acs_col,
                       maf_col,
                       rdata_tsv_fn='local_absolute_rdata_as_tsv',
                       cnp_fig_pkl_fn_col='cnp_figs_pkl',
                       ) -> ReviewDataApp:
        """
        Parameters
        ==========
        sample_info_cols: list[str]
            List of columns in data
        acs_col: str
            Column name in data with path to seg file from alleliccapseg
        maf_col: str
            Column name in data with path to maf file (mutation validator validated maf)
        rdata_tsv_fn: str
            Column name in data with LOCAL path to maf file. Should be predownloaded at set_review_data()
        cnp_fig_pkl_fn_col: str
            Column nae in data with LOCAL path to premade pickled copy number figures
        mut_fig_pkl_fn_col: str
            Column nae in data with LOCAL path to premade pickled mutation figures
        """

        app = ReviewDataApp()

        app.add_component(
            AppComponent(
                'sample-info-component',
                html.Div(
                    children=[html.H1('Data Summary'),
                              dbc.Table.from_dataframe(df=pd.DataFrame())],
                    id='sample-info-component'),
                callback_output=[Output('sample-info-component', 'children')],
                new_data_callback=gen_data_summary_table),
            cols=sample_info_cols)
        
        
        # Add a custom component: below, I add a component that allows you to manually set the 0 and 1 line combs (cgaitools)
        # NOTE: the purity calculated is tau, NOT tau_g. 
        # Use the called purity and tau as inputs to absolute_segforcecall to get tau_g (tau_hat)
        def gen_custom_absolute_component(
                data: GenericData,
                data_id,
                slider_value,  # dash app parameters come first
                cnp_fig_pkl_fn_col):
            data_df = data.df
            r = data_df.loc[data_id]
            cnp_fig = pickle.load(open(r[cnp_fig_pkl_fn_col], "rb"))

            # add 1 and 0 lines
            cnp_fig_with_lines = go.Figure(cnp_fig)
            i = 0
            line_0 = slider_value[0]
            line_1 = slider_value[1]
            line_height = line_0
            step_size = line_1 - line_0
            while line_height < 2:
                line_height = line_0 + (step_size * i)
                cnp_fig_with_lines.add_hline(y=line_height, 
                                             line_dash="dash", 
                                             line_color='black',
                                             line_width=1
                                            )
                i += 1

            purity = 1 - (float(line_0) / float(line_1))
            ploidy = (2 * (1 - line_0) * (1 - purity)) / (purity * line_0) # tau, not tau_g
            return [slider_value,
                    cnp_fig_with_lines, 
                    purity,
                    ploidy]

        # Adding another component to prebuilt dash board
        app.add_component(
            AppComponent(
                'custom-cnp-plot',
                html.Div([html.H2('Custom Copy Number Profile', id='custom-header-cnp-id'),
                          dbc.Row([
                              dbc.Col([dcc.Graph(id='custom-cnp-graph',
                                                 figure={})
                                       ],
                                      md=10),
                              dbc.Col([
                                  dbc.Row(html.Div([html.Div([html.P('Purity: ',
                                                                     style={'display': 'inline'}),
                                                              html.P(0,
                                                                     id='custom-cnp-graph-purity',
                                                                     style={'display': 'inline'})]),
                                                    html.Div([html.P('Ploidy: ',
                                                                     style={'display': 'inline'}),
                                                              html.P(0,
                                                                     id='custom-cnp-graph-ploidy',
                                                                     style={'display': 'inline'})]),
                                                    ])),
                                  dbc.Row(dcc.RangeSlider(id='custom-cnp-slider', min=0.0, max=2.0, step=0.01,
                                                          allowCross=False,
                                                          value=[0.5, 1.0],
                                                          marks={i: f'{i}' for i in range(0, 3, 1)},
                                                          vertical=True,
                                                          tooltip={"placement": "right",
                                                                   "always_visible": True})),
                              ],
                                  md=1),
                          ]),
                          ]),
                new_data_callback=gen_custom_absolute_component,
                internal_callback=gen_custom_absolute_component,
                callback_input=[Input('custom-cnp-slider', 'value')],
                callback_output=[Output('custom-cnp-slider', 'value'),
                                 Output('custom-cnp-graph', 'figure'),
                                 Output('custom-cnp-graph-purity', 'children'),
                                 Output('custom-cnp-graph-ploidy', 'children')
                                 ],
            ),
            cnp_fig_pkl_fn_col='cnp_figs_pkl'
        )
        
        return app

    def set_default_autofill(self):
        self.add_autofill('cnp-plot-button', State('custom-cnp-graph-purity', 'children'), 'Purity')
        self.add_autofill('cnp-plot-button', State('custom-cnp-graph-ploidy', 'children'), 'Ploidy')

    def set_default_review_data_annotations(self):
        self.add_review_data_annotation(
            annot_name='Purity',
            review_data_annotation=DataAnnotation('float', validate_input=validate_purity))
        self.add_review_data_annotation(
            annot_name='Ploidy',
            review_data_annotation=DataAnnotation('float', validate_input=validate_ploidy))

    def set_default_review_data_annotations_app_display(self):
        self.add_review_data_annotations_app_display(annot_name='Purity', app_display_type='number')
        self.add_review_data_annotations_app_display(annot_name='Ploidy', app_display_type='number')

        
        