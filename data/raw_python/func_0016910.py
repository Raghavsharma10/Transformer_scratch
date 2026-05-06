def current_module_name(frame):
    """
    Get name of module of currently running function from inspect/trace
    stack frame.

    Parameters
    ----------
    frame : stack frame
      Stack frame obtained via trace or inspect

    Returns
    -------
    modname : string
      Currently running function module name
    """

    if frame is None:
        return None

    if hasattr(frame.f_globals, '__name__'):
        return frame.f_globals['__name__']
    else:
        mod = inspect.getmodule(frame)
        if mod is None:
            return ''
        else:
            return mod.__name__