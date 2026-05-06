def extract_objects_from_source(self, text, type_filter=None):
    '''Extract object declarations from a text buffer

    Args:
      text (str): Source code to parse
      type_filter (class, optional): Object class to filter results
    Returns:
      List of parsed objects.
    '''
    objects = parse_verilog(text)

    if type_filter:
      objects = [o for o in objects if isinstance(o, type_filter)]

    return objects