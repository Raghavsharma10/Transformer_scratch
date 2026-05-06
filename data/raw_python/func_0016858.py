def _create_axes(filenames, file_dict):
    """ Create a FitsAxes object """

    try:
        # Loop through the file_dictionary, finding the
        # first open FITS file.
        f = iter(f for tup in file_dict.itervalues()
            for f in tup if f is not None).next()
    except StopIteration as e:
        raise (ValueError("No FITS files were found. "
            "Searched filenames: '{f}'." .format(
                f=filenames.values())),
                    None, sys.exc_info()[2])


    # Create a FitsAxes object
    axes = FitsAxes(f[0].header)

    # Scale any axes in degrees to radians
    for i, u in enumerate(axes.cunit):
        if u == 'DEG':
            axes.cunit[i] = 'RAD'
            axes.set_axis_scale(i, np.pi/180.0)

    return axes