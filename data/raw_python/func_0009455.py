def rot90(img):
    '''
    rotate one or multiple grayscale or color images 90 degrees
    '''
    s = img.shape
    if len(s) == 3:
        if s[2] in (3, 4):  # color image
            out = np.empty((s[1], s[0], s[2]), dtype=img.dtype)
            for i in range(s[2]):
                out[:, :, i] = np.rot90(img[:, :, i])
        else:  # mutliple grayscale
            out = np.empty((s[0], s[2], s[1]), dtype=img.dtype)
            for i in range(s[0]):
                out[i] = np.rot90(img[i])
    elif len(s) == 2:  # one grayscale
        out = np.rot90(img)
    elif len(s) == 4 and s[3] in (3, 4):  # multiple color
        out = np.empty((s[0], s[2], s[1], s[3]), dtype=img.dtype)
        for i in range(s[0]):  # for each img
            for j in range(s[3]):  # for each channel
                out[i, :, :, j] = np.rot90(img[i, :, :, j])
    else:
        NotImplemented
    return out