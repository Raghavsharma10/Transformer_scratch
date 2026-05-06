def _hsv_to_rgb(hsvs):
    """Convert Nx3 or Nx4 hsv to rgb"""
    hsvs, n_dim = _check_color_dim(hsvs)
    # In principle, we *might* be able to vectorize this, but might as well
    # wait until a compelling use case appears
    rgbs = list()
    for hsv in hsvs:
        c = hsv[1] * hsv[2]
        m = hsv[2] - c
        hp = hsv[0] / 60
        x = c * (1 - abs(hp % 2 - 1))
        if 0 <= hp < 1:
            r, g, b = c, x, 0
        elif hp < 2:
            r, g, b = x, c, 0
        elif hp < 3:
            r, g, b = 0, c, x
        elif hp < 4:
            r, g, b = 0, x, c
        elif hp < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        rgb = [r + m, g + m, b + m]
        rgbs.append(rgb)
    rgbs = np.array(rgbs, dtype=np.float32)
    if n_dim == 4:
        rgbs = np.concatenate((rgbs, hsvs[:, 3]), axis=1)
    return rgbs