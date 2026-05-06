def load_datafile(name, search_path, codecs=get_codecs(), **kwargs):
    """
    find datafile and load them from codec
    TODO only does the first one
    kwargs:
    default = if passed will return that on failure instead of throwing
    """
    return munge.load_datafile(name, search_path, codecs, **kwargs)