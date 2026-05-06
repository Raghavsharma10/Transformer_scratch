def _isInstalled(fullFname):
    """ Return True if the given file name is located in an
    installed area (versus a user-owned file) """
    if not fullFname: return False
    if not os.path.exists(fullFname): return False
    instAreas = []
    try:
        import site
        instAreas = site.getsitepackages()
    except:
        pass # python 2.6 and lower don't have site.getsitepackages()
    if len(instAreas) < 1:
        instAreas = [ os.path.dirname(os.__file__) ]
    for ia in instAreas:
        if fullFname.find(ia) >= 0:
            return True
    return False