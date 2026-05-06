def load_rf_model():
    """
    Return the UPSILoN random forests classifier.

    The classifier is trained using OGLE and EROS periodic variables
    (Kim et al. 2015).

    Returns
    -------
    clf : sklearn.ensemble.RandomForestClassifier
        The UPSILoN random forests classifier.
    """

    import gzip
    try:
        import cPickle as pickle
    except:
        import pickle

    module_path = dirname(__file__)
    file_path = join(module_path, 'models/rf.model.sub.github.gz')

    # For Python 3.
    if sys.version_info.major >= 3:
        clf = pickle.load(gzip.open(file_path, 'rb'), encoding='latin1')
    # For Python 2.
    else:
        clf = pickle.load(gzip.open(file_path, 'rb'))

    return clf