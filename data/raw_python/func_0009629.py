def extract_objects(self, fname, type_filter=None):
    '''Extract objects from a source file

    Args:
      fname(str): Name of file to read from
      type_filter (class, optional): Object class to filter results
    Returns:
      List of objects extracted from the file.
    '''
    objects = []
    if fname in self.object_cache:
      objects = self.object_cache[fname]
    else:
      with io.open(fname, 'rt', encoding='utf-8') as fh:
        text = fh.read()
        objects = parse_verilog(text)
        self.object_cache[fname] = objects

    if type_filter:
      objects = [o for o in objects if isinstance(o, type_filter)]

    return objects