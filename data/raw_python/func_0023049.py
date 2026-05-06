def _lab_to_rgb(labs):
    """Convert Nx3 or Nx4 lab to rgb"""
    # adapted from BSD-licensed work in MATLAB by Mark Ruzon
    # Based on ITU-R Recommendation BT.709 using the D65
    labs, n_dim = _check_color_dim(labs)

    # Convert Lab->XYZ (silly indexing used to preserve dimensionality)
    y = (labs[:, 0] + 16.) / 116.
    x = (labs[:, 1] / 500.) + y
    z = y - (labs[:, 2] / 200.)
    xyz = np.concatenate(([x], [y], [z]))  # 3xN
    over = xyz > 0.2068966
    xyz[over] = xyz[over] ** 3.
    xyz[~over] = (xyz[~over] - 0.13793103448275862) / 7.787

    # Convert XYZ->LAB
    rgbs = np.dot(_xyz2rgb_norm, xyz).T
    over = rgbs > 0.0031308
    rgbs[over] = 1.055 * (rgbs[over] ** (1. / 2.4)) - 0.055
    rgbs[~over] *= 12.92
    if n_dim == 4:
        rgbs = np.concatenate((rgbs, labs[:, 3]), axis=1)
    rgbs = np.clip(rgbs, 0., 1.)
    return rgbs