def current_function(frame):
    """
    Get reference to currently running function from inspect/trace stack frame.

    Parameters
    ----------
    frame : stack frame
      Stack frame obtained via trace or inspect

    Returns
    -------
    fnc : function reference
      Currently running function
    """

    if frame is None:
        return None

    code = frame.f_code
    # Attempting to extract the function reference for these calls appears
    # to be problematic
    if code.co_name == '__del__' or code.co_name == '_remove' or \
       code.co_name == '_removeHandlerRef':
        return None

    try:
        # Solution follows suggestion at http://stackoverflow.com/a/37099372
        lst = [referer for referer in gc.get_referrers(code)
               if getattr(referer, "__code__", None) is code and
               inspect.getclosurevars(referer).nonlocals.items() <=
               frame.f_locals.items()]
        if lst:
            return lst[0]
        else:
            return None
    except ValueError:
        # inspect.getclosurevars can fail with ValueError: Cell is empty
        return None