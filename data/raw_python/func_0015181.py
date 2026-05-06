def install_dap_from_path(path, update=False, update_allpaths=False, first=True,
                          force=False, nodeps=False, reinstall=False, __ui__=''):
    '''Installs a dap from a given path'''
    will_uninstall = False
    dap_obj = dapi.Dap(path)
    name = dap_obj.meta['package_name']

    if name in get_installed_daps():
        if not update and not reinstall:
            raise DapiLocalError(
                'DAP {name} is already installed. '
                'Run `da pkg list` to see it\'s location, or use --reinstall to ignore this check.'
                .format(name=name))
        elif not update_allpaths and name in get_installed_daps(_install_path()):
            will_uninstall = True
        elif update_allpaths and name in get_installed_daps():
            will_uninstall = True

    if update and update_allpaths:
        install_locations = []
        for pair in get_installed_daps_detailed()[name]:
            install_locations.append(pair['location'])
    else:
        install_locations = [_install_path()]

    # This should not happen unless someone did it on purpose
    for location in install_locations:
        if os.path.isfile(location):
            raise DapiLocalError(
                '{i} is a file, not a directory.'.format(i=_install_path()))

    _dir = tempfile.mkdtemp()

    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    ok = dapi.DapChecker.check(dap_obj)
    logger.setLevel(old_level)

    if not ok:
        raise DapiLocalError('The DAP you want to install has errors, not installing.')

    installed = []
    if first:
        if not force and not _is_supported_here(dap_obj.meta):
            raise DapiLocalError(
                '{0} is not supported on this platform (use --force to suppress this check)'.
                format(name))

        deps = set()
        if 'dependencies' in dap_obj.meta and not nodeps:
            for dep in dap_obj.meta['dependencies']:
                dep = _strip_version_from_dependency(dep)
                if dep not in get_installed_daps():
                    deps |= _get_all_dependencies_of(dep, force=force)
            for dep in deps:
                if dep not in get_installed_daps():
                    installed += install_dap(dep, first=False, __ui__=__ui__)

    dap_obj.extract(_dir)

    if will_uninstall:
        uninstall_dap(name, allpaths=update_allpaths, __ui__=__ui__)

    _dapdir = os.path.join(_dir, name + '-' + dap_obj.meta['version'])

    if not os.path.isdir(_install_path()):
        os.makedirs(_install_path())

    os.mkdir(os.path.join(_dapdir, 'meta'))
    os.rename(os.path.join(_dapdir, 'meta.yaml'),
              os.path.join(_dapdir, 'meta', name + '.yaml'))

    for location in install_locations:
        for f in glob.glob(_dapdir + '/*'):
            dst = os.path.join(location, os.path.basename(f))
            if os.path.isdir(f):
                if not os.path.exists(dst):
                    os.mkdir(dst)
                for src_dir, dirs, files in os.walk(f):
                    dst_dir = src_dir.replace(f, dst)
                    if not os.path.exists(dst_dir):
                        os.mkdir(dst_dir)
                    for file_ in files:
                        src_file = os.path.join(src_dir, file_)
                        dst_file = os.path.join(dst_dir, file_)
                        shutil.copyfile(src_file, dst_file)
            else:
                shutil.copyfile(f, dst)
    try:
        shutil.rmtree(_dir)
    except:
        pass

    return [name] + installed