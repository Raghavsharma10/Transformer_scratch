def combine_regions(container):
    """
    Return a region that is the combination of those specified in the container.
    The container is typically a results instance that comes from argparse.

    Order of construction is: add regions, subtract regions, add circles, subtract circles,
    add polygons, subtract polygons.

    Parameters
    ----------
    container : :class:`AegeanTools.MIMAS.Dummy`
        The regions to be combined.

    Returns
    -------
    region : :class:`AegeanTools.regions.Region`
        The constructed region.
    """
    # create empty region
    region = Region(container.maxdepth)

    # add/rem all the regions from files
    for r in container.add_region:
        logging.info("adding region from {0}".format(r))
        r2 = Region.load(r[0])
        region.union(r2)

    for r in container.rem_region:
        logging.info("removing region from {0}".format(r))
        r2 = Region.load(r[0])
        region.without(r2)


    # add circles
    if len(container.include_circles) > 0:
        for c in container.include_circles:
            circles = np.radians(np.array(c))
            if container.galactic:
                l, b, radii = circles.reshape(3, circles.shape[0]//3)
                ras, decs = galactic2fk5(l, b)
            else:
                ras, decs, radii = circles.reshape(3, circles.shape[0]//3)
            region.add_circles(ras, decs, radii)

    # remove circles
    if len(container.exclude_circles) > 0:
        for c in container.exclude_circles:
            r2 = Region(container.maxdepth)
            circles = np.radians(np.array(c))
            if container.galactic:
                l, b, radii = circles.reshape(3, circles.shape[0]//3)
                ras, decs = galactic2fk5(l, b)
            else:
                ras, decs, radii = circles.reshape(3, circles.shape[0]//3)
            r2.add_circles(ras, decs, radii)
            region.without(r2)

    # add polygons
    if len(container.include_polygons) > 0:
        for p in container.include_polygons:
            poly = np.radians(np.array(p))
            poly = poly.reshape((poly.shape[0]//2, 2))
            region.add_poly(poly)

    # remove polygons
    if len(container.exclude_polygons) > 0:
        for p in container.include_polygons:
            poly = np.array(np.radians(p))
            r2 = Region(container.maxdepth)
            r2.add_poly(poly)
            region.without(r2)

    return region