def get_installed_version_of(name, location=None):
    '''Gets the installed version of the given dap or None if not installed
    Searches in all dirs by default, otherwise in the given one'''
    if location:
        locations = [location]
    else:
        locations = _data_dirs()

    for loc in locations:
        if name not in get_installed_daps(loc):
            continue
        meta = '{d}/meta/{dap}.yaml'.format(d=loc, dap=name)
        data = yaml.load(open(meta), Loader=Loader)
        return str(data['version'])
    return None