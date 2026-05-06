def _get_dependencies_of(name, location=None):
    '''
    Returns list of first level dependencies of the given installed dap
    or dap from Dapi  if not installed
    If a location is specified, this only checks for dap installed in that path
    and return [] if the dap is not located there
    '''
    if not location:
        detailed_dap_list = get_installed_daps_detailed()
        if name not in detailed_dap_list:
            return _get_api_dependencies_of(name)
        location = detailed_dap_list[name][0]['location']

    meta = '{d}/meta/{dap}.yaml'.format(d=location, dap=name)
    try:
        data = yaml.load(open(meta), Loader=Loader)
    except IOError:
        return []
    return data.get('dependencies', [])