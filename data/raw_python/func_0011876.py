def flo(string):
    '''Return the string given by param formatted with the callers locals.'''
    callers_locals = {}
    frame = inspect.currentframe()
    try:
        outerframe = frame.f_back
        callers_locals = outerframe.f_locals
    finally:
        del frame
    return string.format(**callers_locals)