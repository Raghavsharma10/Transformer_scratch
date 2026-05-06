def activate():
    """Install the path-based import components."""

    global PathFinder, FileFinder, ff_path_hook

    path_hook_index = len(sys.path_hooks)
    sys.path_hooks.append(ff_path_hook)
    # Resetting sys.path_importer_cache values,
    # to support the case where we have an implicit package inside an already loaded package,
    # since we need to replace the default importer.
    sys.path_importer_cache.clear()

    # Setting up the meta_path to change package finding logic
    pathfinder_index = len(sys.meta_path)
    sys.meta_path.append(PathFinder)

    return path_hook_index, pathfinder_index