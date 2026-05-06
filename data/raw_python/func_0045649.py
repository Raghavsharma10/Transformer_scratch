def nest_map(control_iter, map_fn):
    """
    Apply ``map_fn`` to the directories defined by ``control_iter``

    For each control file in control_iter, map_fn is called with the directory
    and control file contents as arguments.

    Example::

        >>> list(nest_map(['run1/control.json', 'run2/control.json'],
        ...               lambda d, c: c['run_id']))
        [1, 2]

    :param control_iter: Iterable of paths to JSON control files
    :param function map_fn: Function to run for each control file. It should
            accept two arguments: the directory of the control file and the
            json-decoded contents of the control file.
    :returns: A generator of the results of applying ``map_fn`` to elements in
            ``control_iter``
    """
    def fn(control_path):
        """
        Read the control file, return the result of calling map_fn
        """
        with open(control_path) as fp:
            control = ordered_load(fp)
        dn = os.path.dirname(control_path)
        return map_fn(dn, control)

    mapped = imap(fn, control_iter)
    return mapped