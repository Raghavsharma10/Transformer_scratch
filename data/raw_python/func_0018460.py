def file_tree(start_path):
    """
    Create a nested dictionary that represents the folder structure of `start_path`.

    Liberally adapted from
    http://code.activestate.com/recipes/577879-create-a-nested-dictionary-from-oswalk/
    """
    nested_dirs = {}
    root_dir = start_path.rstrip(os.sep)
    start = root_dir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(root_dir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], nested_dirs)
        parent[folders[-1]] = subdir
    return nested_dirs