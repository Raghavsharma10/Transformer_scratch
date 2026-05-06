def merge_wavelengths(waveset1, waveset2, threshold=1e-12):
    """Return the union of the two sets of wavelengths using
    :func:`numpy.union1d`.

    The merged wavelengths may sometimes contain numbers which are nearly
    equal but differ at levels as small as 1e-14. Having values this
    close together can cause problems down the line. So, here we test
    whether any such small differences are present, with a small
    difference defined as less than ``threshold``. If a small
    difference is present, the lower of the too-close pair is removed.

    Parameters
    ----------
    waveset1, waveset2 : array-like or `None`
        Wavelength values, assumed to be in the same unit already.
        Also see :func:`~synphot.models.get_waveset`.

    threshold : float, optional
        Merged wavelength values are considered "too close together"
        when the difference is smaller than this number.
        The default is 1e-12.

    Returns
    -------
    out_wavelengths : array-like or `None`
        Merged wavelengths. `None` if undefined.

    """
    if waveset1 is None and waveset2 is None:
        out_wavelengths = None
    elif waveset1 is not None and waveset2 is None:
        out_wavelengths = waveset1
    elif waveset1 is None and waveset2 is not None:
        out_wavelengths = waveset2
    else:
        out_wavelengths = np.union1d(waveset1, waveset2)
        delta = out_wavelengths[1:] - out_wavelengths[:-1]
        i_good = np.where(delta > threshold)

        # Remove "too close together" duplicates
        if len(i_good[0]) < delta.size:
            out_wavelengths = np.append(
                out_wavelengths[i_good], out_wavelengths[-1])

    return out_wavelengths