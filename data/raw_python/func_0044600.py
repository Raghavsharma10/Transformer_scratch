def paths_by_depth(paths):
    """Sort list of paths by number of directories in it

    .. todo::

        check if a final '/' is consistently given or ommitted.

    :param iterable paths: iterable containing paths (str)
    :rtype: list
    """
    return sorted(
            paths,
            key=lambda path: path.count(os.path.sep),
            reverse=True
    )