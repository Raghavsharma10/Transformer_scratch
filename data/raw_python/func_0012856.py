def getiddfile(versionid):
    """find the IDD file of the E+ installation"""
    vlist = versionid.split('.')
    if len(vlist) == 1:
        vlist = vlist + ['0', '0']
    elif len(vlist) == 2:
        vlist = vlist + ['0']
    ver_str =  '-'.join(vlist)
    eplus_exe, _  = eppy.runner.run_functions.install_paths(ver_str)
    eplusfolder = os.path.dirname(eplus_exe)
    iddfile = '{}/Energy+.idd'.format(eplusfolder, )
    return iddfile