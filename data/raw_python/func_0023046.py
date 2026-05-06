def _rgb_to_hex(rgbs):
    """Convert rgb to hex triplet"""
    rgbs, n_dim = _check_color_dim(rgbs)
    return np.array(['#%02x%02x%02x' % tuple((255*rgb[:3]).astype(np.uint8))
                     for rgb in rgbs], '|U7')