def explain_features():
    '''print the location of each feature and its version

    if the feature is located inside a git repository, this will also print the git-rev and modified files
    '''
    from ape import tasks
    import featuremonkey
    import os

    featurenames = featuremonkey.get_features_from_equation_file(os.environ['PRODUCT_EQUATION_FILENAME'])

    for featurename in featurenames:
        tasks.explain_feature(featurename)