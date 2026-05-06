def ga4ghImportGlue():
    """
    Call this method before importing a ga4gh module in the scripts dir.
    Otherwise, you will be using the installed package instead of
    the development package.
    Assumes a certain directory structure.
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(path)