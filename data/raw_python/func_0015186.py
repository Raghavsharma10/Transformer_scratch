def _get_api_dependencies_of(name, version='', force=False):
    '''Returns list of first level dependencies of the given dap from Dapi'''
    m, d = _get_metadap_dap(name, version=version)
    # We need the dependencies to install the dap,
    # if the dap is unsupported, raise an exception here
    if not force and not _is_supported_here(d):
        raise DapiLocalError(
            '{0} is not supported on this platform (use --force to suppress this check).'.
            format(name))
    return d.get('dependencies', [])