def get_cube(cube, init=False, pkgs=None, cube_paths=None, config=None,
             backends=None, **kwargs):
    '''
    Dynamically locate and load a metrique cube

    :param cube: name of the cube class to import from given module
    :param init: flag to request initialized instance or uninitialized class
    :param config: config dict to pass on initialization (implies init=True)
    :param pkgs: list of package names to search for the cubes in
    :param cube_path: additional paths to search for modules in (sys.path)
    :param kwargs: additional kwargs to pass to cube during initialization
    '''
    pkgs = pkgs or ['cubes']
    pkgs = [pkgs] if isinstance(pkgs, basestring) else pkgs
    # search in the given path too, if provided
    cube_paths = cube_paths or []
    cube_paths_is_basestring = isinstance(cube_paths, basestring)
    cube_paths = [cube_paths] if cube_paths_is_basestring else cube_paths
    cube_paths = [os.path.expanduser(path) for path in cube_paths]

    # append paths which don't already exist in sys.path to sys.path
    [sys.path.append(path) for path in cube_paths if path not in sys.path]

    pkgs = pkgs + DEFAULT_PKGS
    err = False
    for pkg in pkgs:
        try:
            _cube = _load_cube_pkg(pkg, cube)
        except ImportError as err:
            _cube = None
        if _cube:
            break
    else:
        logger.error(err)
        raise RuntimeError('"%s" not found! %s; %s \n%s)' % (
            cube, pkgs, cube_paths, sys.path))

    if init:
        _cube = _cube(config=config, **kwargs)
    return _cube