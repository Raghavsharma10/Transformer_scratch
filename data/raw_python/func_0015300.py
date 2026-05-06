def get_platforms_set():
    '''Returns set of all possible platforms'''
    # arch and mageia are not in Py2 _supported_dists, so we add them manually
    # Ubuntu adds itself to the list on Ubuntu
    platforms = set([x.lower() for x in platform._supported_dists])
    platforms |= set(['darwin', 'arch', 'mageia', 'ubuntu'])
    return platforms