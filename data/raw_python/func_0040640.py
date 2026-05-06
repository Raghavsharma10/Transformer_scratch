def get_locals(f:Frame) -> str:
    ''' returns a formatted view of the local variables in a frame '''
    return pformat({i:f.f_locals[i] for i in f.f_locals if not i.startswith('__')})