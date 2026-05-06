def prepare_modules(module_paths: list, available: dict) -> dict:
    """
    Scan all paths for external modules and form key-value dict.
    :param module_paths: list of external modules (either python packages or third-party scripts)
    :param available: dict of all registered python modules (can contain python modules from module_paths)
    :return: dict of external modules, where keys are filenames (same as stepnames) and values are the paths
    """
    indexed = {}
    for path in module_paths:
        if not os.path.exists(path) and path not in available:
            err = 'No such path: ' + path
            error(err)
        else:
            for f in os.listdir(path):
                mod_path = join(path, f)
                if f in indexed:
                    warning('Override ' + indexed[f] + ' with ' + mod_path)
                indexed[f] = mod_path
    return indexed