def mark_path(path):
    """
    Wrap given path as relative path relative to top directory.

    Wrapper object will be handled specially in \
    :paramref:`create_cmd_task.parts`.

    :param path: Relative path relative to top directory.

    :return: Wrapper object.
    """
    # If given path is not string,
    # or given path is absolute path.
    if not isinstance(path, str) or os.path.isabs(path):
        # Get error message
        msg = 'Error (2D9ZA): Given path is not relative path: {0}.'.format(
            path
        )

        # Raise error
        raise ValueError(msg)

    # If given path is string,
    # and given path is not absolute path.

    # Wrap given path
    return _ItemWrapper(type='path', item=path)