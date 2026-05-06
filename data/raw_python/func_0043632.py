def convert_halo_to_array_form(halo, ndim):
    """
    Converts the :samp:`{halo}` argument to a :samp:`(ndim, 2)`
    shaped array.

    :type halo: :samp:`None`, :obj:`int`, an :samp:`{ndim}` length sequence
        of :samp:`int` or :samp:`({ndim}, 2)` shaped array
        of :samp:`int`
    :param halo: Halo to be converted to :samp:`({ndim}, 2)` shaped array form.
    :type ndim: :obj:`int`
    :param ndim: Number of dimensions.
    :rtype: :obj:`numpy.ndarray`
    :return: A :samp:`({ndim}, 2)` shaped array of :obj:`numpy.int64` elements.

    Examples::

       >>> convert_halo_to_array_form(halo=2, ndim=4)
       array([[2, 2],
              [2, 2],
              [2, 2],
              [2, 2]])
       >>> convert_halo_to_array_form(halo=[0, 1, 2], ndim=3)
       array([[0, 0],
              [1, 1],
              [2, 2]])
       >>> convert_halo_to_array_form(halo=[[0, 1], [2, 3], [3, 4]], ndim=3)
       array([[0, 1],
              [2, 3],
              [3, 4]])

    """
    dtyp = _np.int64
    if halo is None:
        halo = _np.zeros((ndim, 2), dtype=dtyp)
    elif is_scalar(halo):
        halo = _np.zeros((ndim, 2), dtype=dtyp) + halo
    elif (ndim == 1) and (_np.array(halo).shape == (2,)):
        halo = _np.array([halo, ], copy=True, dtype=dtyp)
    elif len(_np.array(halo).shape) == 1:
        halo = _np.array([halo, halo], dtype=dtyp).T.copy()
    else:
        halo = _np.array(halo, copy=True, dtype=dtyp)

    if halo.shape[0] != ndim:
        raise ValueError(
            "Got halo.shape=%s, expecting halo.shape=(%s, 2)"
            %
            (halo.shape, ndim)
        )

    return halo