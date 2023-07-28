import pandas as pd
import requests
import tqdm
from IPython.display import display, HTML


registered_reviewer_repos = [
    'https://github.com/getzlab/AnnoMate',
    'https://github.com/getzlab/PurityReviewer', 
    'https://github.com/getzlab/PatientReviewer',
    'https://github.com/getzlab/MutationReviewer',
    'https://github.com/getzlab/PhylogicReviewer'
]

catalog_fn = 'catalog.tsv'

def gen_reviewer_catalog():
    repo_data_df = pd.DataFrame(columns=['Repo', 'Type', 'Name', 'Description', 'url'])
    i = 0
    for repo_url in tqdm.tqdm(registered_reviewer_repos, total=len(registered_reviewer_repos)):
        repo_name = repo_url.rsplit('https://github.com/', 1)[-1]
        base_name = repo_name.rsplit('/', 1)[-1]

        request_url = f'https://api.github.com/repos/{repo_name}'
        r = requests.get(request_url)
        res = r.json()
        repo_data_df.loc[i] = {'Repo': repo_name, 'Type': 'Repository', 'Name': repo_name, 'Description': res['description'], 'url': repo_url}


        request_url = f'https://api.github.com/repos/{repo_name}/git/trees/master?recursive=1'
        r = requests.get(request_url)
        res = r.json()


        i += 1
        for file in res["tree"]:
            url = f'{repo_url}/blob/master/{file["path"]}'
            filename = file['path'].rsplit('/')[-1].rsplit('.py')[0]

            if f'{base_name}/Reviewers' in file['path']:
                repo_data_df.loc[i] = {'Repo': repo_name, 'Type': 'Reviewer', 'Name': filename, 'Description': '', 'url': url}
                i += 1

            elif f'{base_name}/AppComponent' in file['path']:
                repo_data_df.loc[i] = {'Repo': repo_name, 'Type': 'AppComponent', 'Name': filename, 'Description': '', 'url': url}
                i += 1
    
    repo_data_df.to_csv(catalog_fn, sep='\t')

def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val.rsplit('/')[-1])


def display_catalog_df(wrap_length=30):
    catalog_data_df = pd.read_csv(catalog_fn, sep='\t', index_col=0)
    catalog_data_df = catalog_data_df[
        ~catalog_data_df['Name'].isin(['__init__', 'AppComponents', 'Reviewers', '__pycache__']) &
        ~catalog_data_df['Name'].str.contains('cpython')
    ]

    catalog_data_df = catalog_data_df.set_index(['Repo', 'Type', 'Name'])
    
    s = catalog_data_df[['url', 'Description']].style.format(
        {'url': make_clickable, 'Description': lambda x: x if not pd.isna(x) else ''}
    ).set_properties(
        **{'background-color': 'white'}
    ).set_table_styles([
        {'selector': 'th',
        'props': [
            # ('border-style', 'solid'),
            # ('border-color', 'black'),
            ('vertical-align','top')
        ]
        }
    ])

    for idx, group_df in catalog_data_df.groupby(level=0):
        s.set_table_styles(
            {group_df.index[0]: [{'selector': '', 'props': 'border-top: 1px solid black;'}]}, 
            overwrite=False, 
            axis=1
        )

    return s
                       
    