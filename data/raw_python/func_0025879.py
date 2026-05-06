def check_exptime(filelist):
    """
    Removes files with EXPTIME==0 from filelist.
    """
    toclose = False
    removed_files = []
    for f in filelist:
        if isinstance(f, str):
            f = fits.open(f)
            toclose = True

        try:
            exptime = f[0].header['EXPTIME']
        except KeyError:
            removed_files.append(f)
            print("Warning:  There are files without keyword EXPTIME")
            continue
        if exptime <= 0:
            removed_files.append(f)
            print("Warning:  There are files with zero exposure time: keyword EXPTIME = 0.0")

    if removed_files != []:
        print("Warning:  Removing the following files from input list")
        for f in removed_files:
            print('\t',f.filename() or "")
    return removed_files