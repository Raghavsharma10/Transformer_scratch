def pkg_available(pkg_name,required=False):
    '''tests if analysis package is available on this machine (e.g., "afni" or "fsl"), and prints an error if ``required``'''
    if pkg_name in pkgs:
        return True
    if required:
        nl.notify('Error: could not find required analysis package %s' % pkg_name,level=nl.level.error)
    return False