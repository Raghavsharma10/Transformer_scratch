def _get_vispy_caller():
    """Helper to get vispy calling function from the stack"""
    records = inspect.stack()
    # first few records are vispy-based logging calls
    for record in records[5:]:
        module = record[0].f_globals['__name__']
        if module.startswith('vispy'):
            line = str(record[0].f_lineno)
            func = record[3]
            cls = record[0].f_locals.get('self', None)
            clsname = "" if cls is None else cls.__class__.__name__ + '.'
            caller = "{0}:{1}{2}({3}): ".format(module, clsname, func, line)
            return caller
    return 'unknown'