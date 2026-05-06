def calculate_tile_shape_for_max_bytes(
    array_shape,
    array_itemsize,
    max_tile_bytes,
    max_tile_shape=None,
    sub_tile_shape=None,
    halo=None
):
    """
    Returns a tile shape :samp:`tile_shape`
    such that :samp:`numpy.product(tile_shape)*numpy.sum({array_itemsize}) <= {max_tile_bytes}`.
    Also, if :samp:`{max_tile_shape} is not None`
    then :samp:`numpy.all(tile_shape <= {max_tile_shape}) is True` and
    if :samp:`{sub_tile_shape} is not None`
    the :samp:`numpy.all((tile_shape % {sub_tile_shape}) == 0) is True`.

    :type array_shape: sequence of :obj:`int`
    :param array_shape: Shape of the array which is to be split into tiles.
    :type array_itemsize: :obj:`int`
    :param array_itemsize: The number of bytes per element of the array to be tiled.
    :type max_tile_bytes: :obj:`int`
    :param max_tile_bytes: The maximum number of bytes for the returned :samp:`tile_shape`.
    :type max_tile_shape: sequence of :obj:`int`
    :param max_tile_shape: Per axis maximum shapes for the returned :samp:`tile_shape`.
    :type sub_tile_shape: sequence of :obj:`int`
    :param sub_tile_shape: The returned :samp:`tile_shape` will be an even multiple
       of this sub-tile shape.
    :type halo: :obj:`int`, sequence of :obj:`int`, or :samp:`(len({array_shape}), 2)`
       shaped :obj:`numpy.ndarray`
    :param halo: How tiles are extended in each axis direction with *halo*
       elements. See :ref:`the-halo-parameter-examples` for meaning of :samp:`{halo}` values.
    :rtype: :obj:`numpy.ndarray`
    :return: A 1D array of shape :samp:`(len(array_shape),)` indicating a *tile shape*
       which will (approximately) uniformly divide the given :samp:`{array_shape}` into
       tiles (sub-arrays).

    Examples::

       >>> from array_split.split import calculate_tile_shape_for_max_bytes
       >>> calculate_tile_shape_for_max_bytes(
       ... array_shape=[512,],
       ... array_itemsize=1,
       ... max_tile_bytes=512
       ... )
       array([512])
       >>> calculate_tile_shape_for_max_bytes(
       ... array_shape=[512,],
       ... array_itemsize=2,  # Doubling the itemsize halves the tile size.
       ... max_tile_bytes=512
       ... )
       array([256])
       >>> calculate_tile_shape_for_max_bytes(
       ... array_shape=[512,],
       ... array_itemsize=1,
       ... max_tile_bytes=512-1  # tile shape will now be halved
       ... )
       array([256])


    """

    logger = _logging.getLogger(__name__ + ".calculate_tile_shape_for_max_bytes")
    logger.debug("calculate_tile_shape_for_max_bytes: enter:")
    logger.debug("array_shape=%s", array_shape)
    logger.debug("array_itemsize=%s", array_itemsize)
    logger.debug("max_tile_bytes=%s", max_tile_bytes)
    logger.debug("max_tile_shape=%s", max_tile_shape)
    logger.debug("sub_tile_shape=%s", sub_tile_shape)
    logger.debug("halo=%s", halo)

    array_shape = _np.array(array_shape, dtype="int64")
    array_itemsize = _np.sum(array_itemsize, dtype="int64")

    if max_tile_shape is None:
        max_tile_shape = _np.array(array_shape, copy=True)
    max_tile_shape = \
        _np.array(_np.minimum(max_tile_shape, array_shape), copy=True, dtype=array_shape.dtype)

    if sub_tile_shape is None:
        sub_tile_shape = _np.ones((len(array_shape),), dtype="int64")

    sub_tile_shape = _np.array(sub_tile_shape, dtype="int64")

    halo = convert_halo_to_array_form(halo=halo, ndim=len(array_shape))

    if _np.any(array_shape < sub_tile_shape):
        raise ValueError(
            "Got array_shape=%s element less than corresponding sub_tile_shape=%s element."
            %
            (
                array_shape,
                sub_tile_shape
            )
        )

    logger.debug("max_tile_shape=%s", max_tile_shape)
    logger.debug("sub_tile_shape=%s", sub_tile_shape)
    logger.debug("halo=%s", halo)
    array_sub_tile_split_shape = ((array_shape - 1) // sub_tile_shape) + 1
    tile_sub_tile_split_shape = array_shape // sub_tile_shape
    if len(tile_sub_tile_split_shape) <= 1:
        tile_sub_tile_split_shape[0] = \
            int(_np.floor(
                (
                    (max_tile_bytes / float(array_itemsize))
                    -
                    _np.sum(halo)
                )
                /
                float(sub_tile_shape[0])
            ))

    tile_sub_tile_split_shape = \
        _np.minimum(
            tile_sub_tile_split_shape,
            max_tile_shape // sub_tile_shape
        )
    logger.debug("Pre loop: tile_sub_tile_split_shape=%s", tile_sub_tile_split_shape)

    current_axis = 0
    while (
        (current_axis < len(tile_sub_tile_split_shape))
        and
        (
            (
                _np.product(tile_sub_tile_split_shape * sub_tile_shape + _np.sum(halo, axis=1))
                *
                array_itemsize
            )
            >
            max_tile_bytes
        )
    ):
        if current_axis < (len(tile_sub_tile_split_shape) - 1):
            tile_sub_tile_split_shape[current_axis] = 1
            tile_sub_tile_split_shape[current_axis] = \
                (
                    max_tile_bytes
                    //
                    (
                        _np.product(
                            tile_sub_tile_split_shape *
                            sub_tile_shape +
                            _np.sum(
                                halo,
                                axis=1))
                        *
                        array_itemsize
                    )
            )
            tile_sub_tile_split_shape[current_axis] = \
                max([1, tile_sub_tile_split_shape[current_axis]])
        else:
            sub_tile_shape_h = sub_tile_shape.copy()
            sub_tile_shape_h[0:current_axis] += _np.sum(halo[0:current_axis, :], axis=1)
            tile_sub_tile_split_shape[current_axis] = \
                int(_np.floor(
                    (
                        (max_tile_bytes / float(array_itemsize))
                        -
                        _np.sum(halo[current_axis]) * _np.product(sub_tile_shape_h[0:current_axis])
                    )
                    /
                    float(_np.product(sub_tile_shape_h))
                ))
        current_axis += 1

    logger.debug("Post loop: tile_sub_tile_split_shape=%s", tile_sub_tile_split_shape)
    tile_shape = _np.minimum(array_shape, tile_sub_tile_split_shape * sub_tile_shape)
    logger.debug("pre cannonicalise tile_shape=%s", tile_shape)

    tile_split_shape = ((array_shape - 1) // tile_shape) + 1
    logger.debug("tile_split_shape=%s", tile_split_shape)

    tile_shape = (((array_sub_tile_split_shape - 1) // tile_split_shape) + 1) * sub_tile_shape
    logger.debug("post cannonicalise tile_shape=%s", tile_shape)

    return tile_shape