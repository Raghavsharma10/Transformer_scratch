def guess_home():
    '''If ``freesurfer_home`` is not set, try to make an intelligent guess at it'''
    global freesurfer_home
    if freesurfer_home != None:
        return True
    # if we already have it in the path, use that
    fv = nl.which('freeview')
    if fv:
        freesurfer_home = parpar_dir(os.path.realpath(fv))
        return True
    for guess_dir in guess_locations:
        if os.path.exists(guess_dir):
            freesurfer_home = guess_dir
            return True
    return False