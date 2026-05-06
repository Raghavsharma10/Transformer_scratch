def get_installed_daps_detailed():
    '''Returns a dictionary with all installed daps and their versions and locations
    First version and location in the dap's list is the one that is preferred'''
    daps = {}
    for loc in _data_dirs():
        s = get_installed_daps(loc)
        for dap in s:
            if dap not in daps:
                daps[dap] = []
            daps[dap].append({'version': get_installed_version_of(dap, loc), 'location': loc})
    return daps