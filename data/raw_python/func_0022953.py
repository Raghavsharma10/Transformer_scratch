def read_from_user(input_type, *args, **kwargs):
    '''
    Helper function to prompt user for input of a specific type 
    e.g. float, str, int 
    Designed to work with both python 2 and 3 
    Yes I know this is ugly.
    '''

    def _read_in(*args, **kwargs):
        while True:
            try: tmp =  raw_input(*args, **kwargs)
            except NameError: tmp =  input(*args, **kwargs)
            try: return input_type(tmp)
            except: print ('Expected type', input_type)

    return _read_in(*args, **kwargs)