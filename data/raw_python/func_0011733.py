def annotations_from_file(filename):
    """Get a list of event annotations from an EDF (European Data Format file
    or EDF+ file, using edflib.

    Args:
      filename: EDF+ file

    Returns:
      list: annotation events, each in the form [start_time, duration, text]
    """
    import edflib
    e = edflib.EdfReader(filename, annotations_mode='all')
    return e.read_annotations()