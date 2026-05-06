def _list_releases():
    '''
    Tries to guess matlab process release version and location path on
    osx machines.

    The paths we will search are in the format:
    /Applications/MATLAB_R[YEAR][VERSION].app/bin/matlab
    We will try the latest version first. If no path is found, None is reutrned.
    '''
    if is_linux():
        base_path = '/usr/local/MATLAB/R%d%s/bin/matlab'
    else:
        # assume mac
        base_path = '/Applications/MATLAB_R%d%s.app/bin/matlab'
    years = range(2050,1990,-1)
    release_letters = ('h', 'g', 'f', 'e', 'd', 'c', 'b', 'a')
    for year in years:
        for letter in release_letters:
            release = 'R%d%s' % (year, letter)
            matlab_path = base_path % (year, letter)
            if os.path.exists(matlab_path):
                yield (release, matlab_path)