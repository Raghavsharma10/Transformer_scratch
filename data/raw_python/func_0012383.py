def batch_process_data(file_roots, **kwargs):
    """Process output from many nested sampling runs in parallel with optional
    error handling and caching.

    The result can be cached using the 'save_name', 'save' and 'load' kwargs
    (by default this is not done). See save_load_result docstring for more
    details.

    Remaining kwargs passed to parallel_utils.parallel_apply (see its
    docstring for more details).

    Parameters
    ----------
    file_roots: list of strs
        file_roots for the runs to load.
    base_dir: str, optional
        path to directory containing files.
    process_func: function, optional
        function to use to process the data.
    func_kwargs: dict, optional
        additional keyword arguments for process_func.
    errors_to_handle: error or tuple of errors, optional
        which errors to catch when they occur in processing rather than
        raising.
    save_name: str or None, optional
        See nestcheck.io_utils.save_load_result.
    save: bool, optional
        See nestcheck.io_utils.save_load_result.
    load: bool, optional
        See nestcheck.io_utils.save_load_result.
    overwrite_existing: bool, optional
        See nestcheck.io_utils.save_load_result.

    Returns
    -------
    list of ns_run dicts
        List of nested sampling runs in dict format (see the module
        docstring for more details).
    """
    base_dir = kwargs.pop('base_dir', 'chains')
    process_func = kwargs.pop('process_func', process_polychord_run)
    func_kwargs = kwargs.pop('func_kwargs', {})
    func_kwargs['errors_to_handle'] = kwargs.pop('errors_to_handle', ())
    data = nestcheck.parallel_utils.parallel_apply(
        process_error_helper, file_roots, func_args=(base_dir, process_func),
        func_kwargs=func_kwargs, **kwargs)
    # Sort processed runs into the same order as file_roots (as parallel_apply
    # does not preserve order)
    data = sorted(data,
                  key=lambda x: file_roots.index(x['output']['file_root']))
    # Extract error information and print
    errors = {}
    for i, run in enumerate(data):
        if 'error' in run:
            try:
                errors[run['error']].append(i)
            except KeyError:
                errors[run['error']] = [i]
    for error_name, index_list in errors.items():
        message = (error_name + ' processing ' + str(len(index_list)) + ' / '
                   + str(len(file_roots)) + ' files')
        if len(index_list) != len(file_roots):
            message += ('. Roots with errors have (zero based) indexes: '
                        + str(index_list))
        print(message)
    # Return runs which did not have errors
    return [run for run in data if 'error' not in run]