def sort(amend: bool = False, stage: bool = False):
    """
    Runs iSort (https://pypi.python.org/pypi/isort)

    Args:
        amend: whether or not to commit results
        stage: whether or not to stage changes
    """
    _sort(amend, stage)