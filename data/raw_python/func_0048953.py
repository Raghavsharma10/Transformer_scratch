def stack_frame_info(stacklevel):
    '''
    Return a named tuple with information about the given stack frame:
        - filename
        - line_number
        - module_name
        - function_name

    stacklevel: How far up the stack to look. 1 means the immediate caller, 2
      its caller, and so on.
    '''
    import inspect

    if stacklevel < 1:
        raise ValueError('A stacklevel less than 1 is pointless')

    frame, filename, line_number, function_name, _, _ = inspect.stack()[stacklevel]
    module = inspect.getmodule(frame) # it is possible for getmodule to return None
    if module is not None:
        module_name = module.__name__
    else:
        module_name = ""

    return Where(
        filename=filename,
        line_number=line_number,
        module_name=module_name,
        function_name=function_name
    )