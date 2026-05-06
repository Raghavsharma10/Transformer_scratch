def open():
    global _MATLAB_RELEASE
    '''Opens MATLAB using specified connection (or DCOM+ protocol on Windows)where matlab_location  '''
    if is_win:
        ret = MatlabConnection()
        ret.open()
        return ret
    else:
        if settings.MATLAB_PATH != 'guess':
            matlab_path = settings.MATLAB_PATH + '/bin/matlab'
        elif _MATLAB_RELEASE != 'latest':
            matlab_path = discover_location(_MATLAB_RELEASE)
        else:
            # Latest release is found in __init__.by, i.e. higher logical level
            raise MatlabReleaseNotFound('Please select a matlab release or set its location.')
        try:
            ret = MatlabConnection(matlab_path)
            ret.open()
        except Exception:
            #traceback.print_exc(file=sys.stderr)
            raise MatlabReleaseNotFound('Could not open matlab, is it in %s?' % matlab_path)
        return ret