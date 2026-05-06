def install_dap(name, version='', update=False, update_allpaths=False, first=True,
                force=False, nodeps=False, reinstall=False, __ui__=''):
    '''Install a dap from dapi
    If update is True, it will remove previously installed daps of the same name'''
    m, d = _get_metadap_dap(name, version)
    if update:
        available = d['version']
        current = get_installed_version_of(name)
        if not current:
            raise DapiLocalError('Cannot update not yet installed DAP.')
        if dapver.compare(available, current) <= 0:
            return []
    path, remove_dir = download_dap(name, d=d)

    ret = install_dap_from_path(path, update=update, update_allpaths=update_allpaths, first=first,
                                force=force, nodeps=nodeps, reinstall=reinstall, __ui__=__ui__)

    try:
        if remove_dir:
            shutil.rmtree(os.dirname(path))
        else:
            os.remove(path)
    except:
        pass

    return ret