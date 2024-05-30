import pandas as pd
from AnnoMate.Reviewers.ExampleReviewer import ExampleReviewer

def test_reviewer():
    fn = '../tutorial_notebooks/example_data/AnnoMate_Tutorial/data_to_review_example.tsv'
    df = pd.read_csv(fn, sep='\t')
    df = df.set_index('sample_id')
    assert df.shape[0] > 0

    my_reviewer = ExampleReviewer()
    output_pkl_path = './example_reviewer_data'
    my_reviewer.set_review_data(data_path=output_pkl_path, 
                                description='Intro to reviewers review session part 2',
                                sample_df=df,
                                preprocessing_str='Testing preprocessing')
    
    assert my_reviewer.review_data_interface.mh.metadata['freeze_data'] == False
    # test state of the reviewer on instantiation
    assert list(my_reviewer.list_data_attributes()) == ['index', 'description', 'annot_col_config_dict', 'annot_df', 'history_df', 'df']
    assert my_reviewer.get_data_attribute('df').equals(df)
    
    
