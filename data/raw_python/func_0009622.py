def save_array_types(self, fname):
    '''Save array type registry to a file
    
    Args:
      fname (str): Name of file to save array database to
    '''
    type_defs = {'arrays': sorted(list(self.array_types))}
    with open(fname, 'wt') as fh:
      pprint(type_defs, stream=fh)