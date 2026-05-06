def os_tree(directory, enable_scandir=False):
    """
    Return a directories contents as a dictionary hierarchy.

    .. code:: python

        reusables.os_tree(".")
        # {'doc': {'build': {'doctrees': {},
        #                   'html': {'_sources': {}, '_static': {}}},
        #         'source': {}},
        #  'reusables': {'__pycache__': {}},
        #  'test': {'__pycache__': {}, 'data': {}}}


    :param directory: path to directory to created the tree of.
    :param enable_scandir: on python < 3.5 enable external scandir package
    :return: dictionary of the directory
    """
    if not os.path.exists(directory):
        raise OSError("Directory does not exist")
    if not os.path.isdir(directory):
        raise OSError("Path is not a directory")

    full_list = []
    for root, dirs, files in _walk(directory, enable_scandir=enable_scandir):
        full_list.extend([os.path.join(root, d).lstrip(directory) + os.sep
                          for d in dirs])
    tree = {os.path.basename(directory): {}}
    for item in full_list:
        separated = item.split(os.sep)
        is_dir = separated[-1:] == ['']
        if is_dir:
            separated = separated[:-1]
        parent = tree[os.path.basename(directory)]
        for index, path in enumerate(separated):
            if path in parent:
                parent = parent[path]
                continue
            else:
                parent[path] = dict()
            parent = parent[path]
    return tree