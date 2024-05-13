from AnnoMate.Reviewers.ExampleReviewer import ExampleReviewer

def f():
    return 3


def test_reviewer():
    my_reviewer = ExampleReviewer()

    # test state of the reviewer on instantiation
    assert my_reviewer.get_annot_df() is None


    fn = '../tutorial_notebooks/example_data/AnnoMate_Tutorial/data_to_review_example.tsv'
    df = pd.read_csv(fn, sep='\t')
    df = df.set_index('sample_id')
    df.head()

    assert df.shape[0] > 0
    