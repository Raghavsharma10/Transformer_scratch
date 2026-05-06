def _array_clip_val(val):
    """Helper to turn val into array and clip between 0 and 1"""
    val = np.array(val)
    if val.max() > 1 or val.min() < 0:
        logger.warning('value will be clipped between 0 and 1')
    val[...] = np.clip(val, 0, 1)
    return val