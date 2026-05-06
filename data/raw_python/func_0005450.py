def wrap_paths(paths):
    # type: (list[str]) -> str
    """ Put quotes around all paths and join them with space in-between. """
    if isinstance(paths, string_types):
        raise ValueError(
            "paths cannot be a string. "
            "Use array with one element instead."
        )
    return ' '.join('"' + path + '"' for path in paths)