def parse_path(f1, f2):

    """Parse two input arguments and return two lists of file names"""

    import glob

    # if second argument is missing or is a wild card, point it
    # to the current directory
    f2 = f2.strip()
    if f2 == '' or f2 == '*':
        f2 = './'

    # if the first argument is a directory, use all GEIS files
    if os.path.isdir(f1):
        f1 = os.path.join(f1, '*.??h')
    list1 = glob.glob(f1)
    list1 = [name for name in list1 if name[-1] == 'h' and name[-4] == '.']

    # if the second argument is a directory, use file names in the
    # first argument to construct file names, i.e.
    # abc.xyh will be converted to abc_xyf.fits
    if os.path.isdir(f2):
        list2 = []
        for file in list1:
            name = os.path.split(file)[-1]
            fitsname = name[:-4] + '_' + name[-3:-1] + 'f.fits'
            list2.append(os.path.join(f2, fitsname))
    else:
        list2 = [s.strip() for s in f2.split(",")]

    if list1 == [] or list2 == []:
        err_msg = ""
        if list1 == []:
            err_msg += "Input files `{:s}` not usable/available. ".format(f1)

        if list2 == []:
            err_msg += "Input files `{:s}` not usable/available. ".format(f2)

        raise IOError(err_msg)

    else:
        return list1, list2