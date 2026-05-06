def register_array_types_from_sources(self, source_files):
    '''Add array type definitions from a file list to internal registry

    Args:
      source_files (list of str): Files to parse for array definitions
    '''
    for fname in source_files:
      if is_vhdl(fname):
        self._register_array_types(self.extract_objects(fname))