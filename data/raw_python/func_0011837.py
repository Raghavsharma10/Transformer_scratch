def find_datafile(name, search_path, codecs=get_codecs()):
    """
    find all matching data files in search_path
    search_path: path of directories to load from
    codecs: allow to override from list of installed
    returns array of tuples (codec_object, filename)
    """
    return munge.find_datafile(name, search_path, codecs)