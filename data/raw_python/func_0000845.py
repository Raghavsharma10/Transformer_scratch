def import_metadata(module_paths):
    """Import all the given modules"""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    modules = []
    try:
        for path in module_paths:
            modules.append(import_module(path))
    except ImportError as e:
        err = RuntimeError('Could not import {}: {}'.format(path, str(e)))
        raise_from(err, e)
    return modules