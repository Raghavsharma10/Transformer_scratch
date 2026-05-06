def writeIslandContours(filename, catalog, fmt='reg'):
    """
    Write an output file in ds9 .reg format that outlines the boundaries of each island.

    Parameters
    ----------
    filename : str
        Filename to write.

    catalog : list
        List of sources. Only those of type :class:`AegeanTools.models.IslandSource` will have contours drawn.

    fmt : str
        Output format type. Currently only 'reg' is supported (default)

    Returns
    -------
    None

    See Also
    --------
    :func:`AegeanTools.catalogs.writeIslandBoxes`
    """
    if fmt != 'reg':
        log.warning("Format {0} not yet supported".format(fmt))
        log.warning("not writing anything")
        return

    out = open(filename, 'w')
    print("#Aegean island contours", file=out)
    print("#AegeanTools.catalogs version {0}-({1})".format(__version__, __date__), file=out)
    line_fmt = 'image;line({0},{1},{2},{3})'
    text_fmt = 'fk5; text({0},{1}) # text={{{2}}}'
    mas_fmt = 'image; line({1},{0},{3},{2}) #color = yellow'
    x_fmt = 'image; point({1},{0}) # point=x'
    for c in catalog:
        contour = c.contour
        if len(contour) > 1:
            for p1, p2 in zip(contour[:-1], contour[1:]):
                print(line_fmt.format(p1[1] + 0.5, p1[0] + 0.5, p2[1] + 0.5, p2[0] + 0.5), file=out)
            print(line_fmt.format(contour[-1][1] + 0.5, contour[-1][0] + 0.5, contour[0][1] + 0.5,
                                          contour[0][0] + 0.5), file=out)
        # comment out lines that have invalid ra/dec (WCS problems)
        if np.nan in [c.ra, c.dec]:
            print('#', end=' ', file=out)
        # some islands may not have anchors because they don't have any contours
        if len(c.max_angular_size_anchors) == 4:
            print(text_fmt.format(c.ra, c.dec, c.island), file=out)
            print(mas_fmt.format(*[a + 0.5 for a in c.max_angular_size_anchors]), file=out)
        for p1, p2 in c.pix_mask:
            # DS9 uses 1-based instead of 0-based indexing
            print(x_fmt.format(p1 + 1, p2 + 1), file=out)
    out.close()
    return