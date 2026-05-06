def timeseries_from_file(filename):
    """Load a multi-channel Timeseries from any file type supported by `biosig`

    Supported file formats include EDF/EDF+, BDF/BDF+, EEG, CNT and GDF.
    Full list is here: http://pub.ist.ac.at/~schloegl/biosig/TESTED

    For EDF, EDF+, BDF and BDF+ files, we will use python-edf 
    if it is installed, otherwise will fall back to python-biosig.

    Args: 
      filename

    Returns: 
      Timeseries
    """
    if not path.isfile(filename):
        raise Error("file not found: '%s'" % filename)
    is_edf_bdf = (filename[-4:].lower() in ['.edf', '.bdf'])
    if is_edf_bdf:
        try:
            import edflib
            return _load_edflib(filename)
        except ImportError:
            print('python-edf not installed. trying python-biosig instead...')
    try:
        import biosig
        return _load_biosig(filename)
    except ImportError:
        message = (
            """To load timeseries from file, ensure python-biosig is installed
            e.g. on Ubuntu or Debian type `apt-get install python-biosig`
            or get it from http://biosig.sf.net/download.html""")
        if is_edf_bdf:
            message += """\n(For EDF/BDF files, can instead install python-edf:
                       https://bitbucket.org/cleemesser/python-edf/ )"""
        raise Error(message)