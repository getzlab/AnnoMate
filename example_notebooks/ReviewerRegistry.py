import pandas as pd
import requests


registered_reviewer_repos = [
    'https://github.com/getzlab/PurityReviewers', 
    'https://github.com/getzlab/PatientReviewer',
    'https://github.com/getzlab/MutationReviewer',
    'https://github.com/getzlab/PhylogicReviewer'
]

registry_fn = 'registry.tsv'

def gen_reviewer_registry():
    repo_data_df = pd.DataFrame(columns=['Repo', 'Type', 'Name', 'Description', 'url'])
    i = 0
    for repo_url in registered_reviewer_repos:
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
    
    repo_data_df.to_csv(repo_data_fn, sep='\t')

def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val.rsplit('/')[-1])


def display_registry_df():
    repo_data_df = pd.read_csv(registry_fn, sep='\t', index_col=0).set_index(['Repo', 'Type', 'Name'])
    return repo_data_df.style.format({'url': make_clickable, 'Description': lambda x: x if not pd.isna(x) else ''}) 
