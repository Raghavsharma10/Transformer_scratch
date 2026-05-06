def _rgb_to_hsv(rgbs):
    """Convert Nx3 or Nx4 rgb to hsv"""
    rgbs, n_dim = _check_color_dim(rgbs)
    hsvs = list()
    for rgb in rgbs:
        rgb = rgb[:3]  # don't use alpha here
        idx = np.argmax(rgb)
        val = rgb[idx]
        c = val - np.min(rgb)
        if c == 0:
            hue = 0
            sat = 0
        else:
            if idx == 0:  # R == max
                hue = ((rgb[1] - rgb[2]) / c) % 6
            elif idx == 1:  # G == max
                hue = (rgb[2] - rgb[0]) / c + 2
            else:  # B == max
                hue = (rgb[0] - rgb[1]) / c + 4
            hue *= 60
            sat = c / val
        hsv = [hue, sat, val]
        hsvs.append(hsv)
    hsvs = np.array(hsvs, dtype=np.float32)
    if n_dim == 4:
        hsvs = np.concatenate((hsvs, rgbs[:, 3]), axis=1)
    return hsvs