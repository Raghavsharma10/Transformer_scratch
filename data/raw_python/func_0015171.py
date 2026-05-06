def _get_metadap_dap(name, version=''):
    '''Return data for dap of given or latest version.'''
    m = metadap(name)
    if not m:
        raise DapiCommError('DAP {dap} not found.'.format(dap=name))
    if not version:
        d = m['latest_stable'] or m['latest']
        if d:
            d = data(d)
    else:
        d = dap(name, version)
        if not d:
            raise DapiCommError(
                'DAP {dap} doesn\'t have version {version}.'.format(dap=name, version=version))
    return m, d