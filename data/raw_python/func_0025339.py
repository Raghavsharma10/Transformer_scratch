def load_tree_object(filename):
    """
    Load scikit-learn decision tree ensemble object from file.
    
    Parameters
    ----------
    filename : str
        Name of the pickle file containing the tree object.
    
    Returns
    -------
    tree ensemble object
    """
    with open(filename) as file_obj:
        tree_ensemble_obj = pickle.load(file_obj)
    return tree_ensemble_obj