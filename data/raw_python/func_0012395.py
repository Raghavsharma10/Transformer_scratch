def select_tqdm():
    """If running in a jupyter notebook, then returns tqdm_notebook.
    Otherwise returns a regular tqdm progress bar.

    Returns
    -------
    progress: function
    """
    try:
        progress = tqdm.tqdm_notebook
        assert get_ipython().has_trait('kernel')
    except (NameError, AssertionError):
        progress = tqdm.tqdm
    return progress