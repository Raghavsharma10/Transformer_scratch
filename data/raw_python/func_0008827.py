def reg2mim(regfile, mimfile, maxdepth):
    """
    Parse a DS9 region file and write a MIMAS region (.mim) file.

    Parameters
    ----------
    regfile : str
        DS9 region (.reg) file.

    mimfile : str
        MIMAS region (.mim) file.

    maxdepth : str
        Depth/resolution of the region file.

    """
    logging.info("Reading regions from {0}".format(regfile))
    lines = (l for l in open(regfile, 'r') if not l.startswith('#'))
    poly = []
    circles = []
    for line in lines:
        if line.startswith('box'):
            poly.append(box2poly(line))
        elif line.startswith('circle'):
            circles.append(circle2circle(line))
        elif line.startswith('polygon'):
            logging.warning("Polygons break a lot, but I'll try this one anyway.")
            poly.append(poly2poly(line))
        else:
            logging.warning("Not sure what to do with {0}".format(line[:-1]))
    container = Dummy(maxdepth=maxdepth)
    container.include_circles = circles
    container.include_polygons = poly

    region = combine_regions(container)
    save_region(region, mimfile)
    return