def load_array_types(self, fname):
    '''Load file of previously extracted data types
    
    Args:
      fname (str): Name of file to load array database from
    '''
    type_defs = ''
    with open(fname, 'rt') as fh:
      type_defs = fh.read()

    try:
      type_defs = ast.literal_eval(type_defs)
    except SyntaxError:
      type_defs = {}

    self._add_array_types(type_defs)