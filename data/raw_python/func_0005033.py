def _pad_nochord(target, axis=-1):
    '''Pad a chord annotation with no-chord flags.

    Parameters
    ----------
    target : np.ndarray
        the input data

    axis : int
        the axis along which to pad

    Returns
    -------
    target_pad
        `target` expanded by 1 along the specified `axis`.
        The expanded dimension will be 0 when `target` is non-zero
        before padding, and 1 otherwise.
    '''
    ncmask = ~np.max(target, axis=axis, keepdims=True)

    return np.concatenate([target, ncmask], axis=axis)