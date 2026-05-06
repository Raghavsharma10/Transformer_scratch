def opener_from_zipfile(zipfile):
    """
    Returns a function that will open a file in a zipfile by name.

    For Python3 compatibility, the raw file will be converted to text.
    """

    def opener(filename):
        inner_file = zipfile.open(filename)
        if PY3:
            from io import TextIOWrapper
            return TextIOWrapper(inner_file)
        else:
            return inner_file

    return opener