def pep8(amend: bool = False, stage: bool = False):
    """
    Runs Pyup's Safety tool (https://pyup.io/safety/)

    Args:
        amend: whether or not to commit results
        stage: whether or not to stage changes
    """
    _pep8(amend, stage)