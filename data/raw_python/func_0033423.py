def _load_cube_pkg(pkg, cube):
    '''
    NOTE: all items in fromlist must be strings
    '''
    try:
        # First, assume the cube module is available
        # with the name exactly as written
        fromlist = map(str, [cube])
        mcubes = __import__(pkg, fromlist=fromlist)
        return getattr(mcubes, cube)
    except AttributeError:
        # if that fails, try to guess the cube module
        # based on cube 'standard naming convention'
        # ie, group_cube -> from group.cube import CubeClass
        _pkg, _mod, _cls = cube_pkg_mod_cls(cube)
        fromlist = map(str, [_cls])
        mcubes = __import__('%s.%s.%s' % (pkg, _pkg, _mod),
                            fromlist=fromlist)
        return getattr(mcubes, _cls)