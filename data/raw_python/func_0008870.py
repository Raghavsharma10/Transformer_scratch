def writeIslandBoxes(filename, catalog, fmt):
    """
    Write an output file in ds9 .reg, or kvis .ann format that contains bounding boxes for all the islands.

    Parameters
    ----------
    filename : str
        Filename to write.

    catalog : list
        List of sources. Only those of type :class:`AegeanTools.models.IslandSource` will have contours drawn.

    fmt : str
        Output format type. Currently only 'reg' and 'ann' are supported. Default = 'reg'.

    Returns
    -------
    None

    See Also
    --------
    :func:`AegeanTools.catalogs.writeIslandContours`
    """
    if fmt not in ['reg', 'ann']:
        log.warning("Format not supported for island boxes{0}".format(fmt))
        return  # fmt not supported

    out = open(filename, 'w')
    print("#Aegean Islands", file=out)
    print("#Aegean version {0}-({1})".format(__version__, __date__), file=out)

    if fmt == 'reg':
        print("IMAGE", file=out)
        box_fmt = 'box({0},{1},{2},{3}) #{4}'
    else:
        print("COORD P", file=out)
        box_fmt = 'box P {0} {1} {2} {3} #{4}'

    for c in catalog:
        # x/y swap for pyfits/numpy translation
        ymin, ymax, xmin, xmax = c.extent
        # +1 for array/image offset
        xcen = (xmin + xmax) / 2.0 + 1
        # + 0.5 in each direction to make lines run 'between' DS9 pixels
        xwidth = xmax - xmin + 1
        ycen = (ymin + ymax) / 2.0 + 1
        ywidth = ymax - ymin + 1
        print(box_fmt.format(xcen, ycen, xwidth, ywidth, c.island), file=out)
    out.close()
    return