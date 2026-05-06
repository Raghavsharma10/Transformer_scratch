def applyColorMap(gray, cmap='flame'):
    '''
    like cv2.applyColorMap(im_gray, cv2.COLORMAP_*) but with different color maps
    '''
    # TODO:implement more cmaps
    if cmap != 'flame':
        raise NotImplemented
    # TODO: make better
    mx = 256  # if gray.dtype==np.uint8 else 65535
    lut = np.empty(shape=(256, 3))
    cmap = (
        # taken from pyqtgraph GradientEditorItem
        (0, (0, 0, 0)),
        (0.2, (7, 0, 220)),
        (0.5, (236, 0, 134)),
        (0.8, (246, 246, 0)),
        (1.0, (255, 255, 255))
    )
    # build lookup table:
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * mx)
        for i in range(3):
            lut[lastval:val, i] = np.linspace(
                lastcol[i], col[i], val - lastval)

        lastcol = col
        lastval = val

    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, i])
    return out