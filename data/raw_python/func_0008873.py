def norm_dist(src1, src2):
    """
    Calculate the normalised distance between two sources.
    Sources are elliptical Gaussians.

    The normalised distance is calculated as the GCD distance between the centers,
    divided by quadrature sum of the radius of each ellipse along a line joining the two ellipses.

    For ellipses that touch at a single point, the normalized distance will be 1/sqrt(2).

    Parameters
    ----------
    src1, src2 : object
        The two positions to compare. Objects must have the following parameters: (ra, dec, a, b, pa).

    Returns
    -------
    dist: float
        The normalised distance.

    """
    if np.all(src1 == src2):
        return 0
    dist = gcd(src1.ra, src1.dec, src2.ra, src2.dec) # degrees

    # the angle between the ellipse centers
    phi = bear(src1.ra, src1.dec, src2.ra, src2.dec) # Degrees
    # Calculate the radius of each ellipse along a line that joins their centers.
    r1 = src1.a*src1.b / np.hypot(src1.a * np.sin(np.radians(phi - src1.pa)),
                                  src1.b * np.cos(np.radians(phi - src1.pa)))
    r2 = src2.a*src2.b / np.hypot(src2.a * np.sin(np.radians(180 + phi - src2.pa)),
                                  src2.b * np.cos(np.radians(180 + phi - src2.pa)))
    R = dist / (np.hypot(r1, r2) / 3600)
    return R