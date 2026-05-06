def process_error_helper(root, base_dir, process_func, errors_to_handle=(),
                         **func_kwargs):
    """Wrapper which applies process_func and handles some common errors so one
    bad run does not spoil the whole batch.

    Useful errors to handle include:

    OSError: if you are not sure if all the files exist
    AssertionError: if some of the many assertions fail for known reasons;
    for example is there are occasional problems decomposing runs into threads
    due to limited numerical precision in logls.

    Parameters
    ----------
    root: str
        File root.
    base_dir: str
        Directory containing file.
    process_func: func
        Function for processing file.
    errors_to_handle: error type or tuple of error types
        Errors to catch without throwing an exception.
    func_kwargs: dict
        Kwargs to pass to process_func.

    Returns
    -------
    run: dict
        Nested sampling run dict (see the module docstring for more
        details) or, if an error occured, a dict containing its type
        and the file root.
    """
    try:
        return process_func(root, base_dir, **func_kwargs)
    except errors_to_handle as err:
        run = {'error': type(err).__name__,
               'output': {'file_root': root}}
        return run