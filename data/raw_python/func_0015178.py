def get_installed_daps(location=None, skip_distro=False):
    '''Returns a set of all installed daps
    Either in the given location or in all of them'''
    if location:
        locations = [location]
    else:
        locations = _data_dirs()
    s = set()
    for loc in locations:
        if skip_distro and loc == DISTRO_DIRECTORY:
            continue
        g = glob.glob('{d}/meta/*.yaml'.format(d=loc))
        for meta in g:
            s.add(meta.split('/')[-1][:-len('.yaml')])
    return s