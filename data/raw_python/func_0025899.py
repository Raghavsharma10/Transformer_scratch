def bitmask2mask(bitmask, ignore_bits, good_mask_value=1, dtype=np.bool_):
    """
    bitmask2mask(bitmask, ignore_bits, good_mask_value=1, dtype=numpy.bool\_)
    Interprets an array of bit flags and converts it to a "binary" mask array.
    This function is particularly useful to convert data quality arrays to
    binary masks.

    Parameters
    ----------
    bitmask : numpy.ndarray
        An array of bit flags. Values different from zero are interpreted as
        "bad" values and values equal to zero are considered as "good" values.
        However, see `ignore_bits` parameter on how to ignore some bits
        in the `bitmask` array.

    ignore_bits : int, str, None
        An integer bit mask, `None`, or a comma- or '+'-separated
        string list of integer bit values that indicate what bits in the
        input `bitmask` should be *ignored* (i.e., zeroed). If `ignore_bits`
        is a `str` and if it is prepended with '~', then the meaning
        of `ignore_bits` parameters will be reversed: now it will be
        interpreted as a list of bits to be *used* (or *not ignored*) when
        deciding what elements of the input `bitmask` array are "bad".

        The `ignore_bits` parameter is the integer sum of all of the bit
        values from the input `bitmask` array that should be considered
        "good" when creating the output binary mask. For example, if
        values in the `bitmask` array can be combinations
        of 1, 2, 4, and 8 flags and one wants to consider that
        values having *only* bit flags 2 and/or 4 as being "good",
        then `ignore_bits` should be set to 2+4=6. Then a `bitmask` element
        having values 2,4, or 6 will be considered "good", while an
        element with a value, e.g., 1+2=3, 4+8=12, etc. will be interpreted
        as "bad".

        Alternatively, one can enter a comma- or '+'-separated list
        of integer bit flags that should be added to obtain the
        final "good" bits. For example, both ``4,8`` and ``4+8``
        are equivalent to setting `ignore_bits` to 12.

        See :py:func:`interpret_bits_value` for examples.

        | Setting `ignore_bits` to `None` effectively will interpret
          all `bitmask` elements as "good" regardless of their value.

        | Setting `ignore_bits` to 0 effectively will assume that all
          non-zero elements in the input `bitmask` array are to be
          interpreted as "bad".

        | In order to reverse the meaning of the `ignore_bits`
          parameter from indicating bits in the values of `bitmask`
          elements that should be ignored when deciding which elements
          are "good" (these are the elements that are zero after ignoring
          `ignore_bits`), to indicating the bits should be used
          exclusively in deciding whether a `bitmask` element is "good",
          prepend '~' to the string value. For example, in order to use
          **only** (or **exclusively**) flags 4 and 8 (2nd and 3rd bits)
          in the values of the input `bitmask` array when deciding whether
          or not that element is "good", set `ignore_bits` to ``~4+8``,
          or ``~4,8 To obtain the same effect with an `int` input value
          (except for 0), enter -(4+8+1)=-9. Following this convention,
          a `ignore_bits` string value of ``'~0'`` would be equivalent to
          setting ``ignore_bits=None``.

    good_mask_value : int, bool (Default = 1)
        This parameter is used to derive the values that will be assigned to
        the elements in the output `mask` array that correspond to the "good"
        flags (that are 0 after zeroing bits specified by `ignore_bits`)
        in the input `bitmask` array. When `good_mask_value` is non-zero or
        `True` then values in the output mask array corresponding to "good"
        bit flags in `bitmask` will be 1 (or `True` if `dtype` is `bool`) and
        values of corresponding to "bad" flags will be 0. When
        `good_mask_value` is zero or `False` then values in the output mask
        array corresponding to "good" bit flags in `bitmask` will be 0
        (or `False` if `dtype` is `bool`) and values of corresponding
        to "bad" flags will be 1.

    dtype : data-type (Default = numpy.uint8)
        The desired data-type for the output binary mask array.

    Returns
    -------
    mask : numpy.ndarray
        Returns an array whose elements can have two possible values,
        e.g., 1 or 0 (or `True` or `False` if `dtype` is `bool`) according to
        values of to the input `bitmask` elements, `ignore_bits` parameter,
        and the `good_mask_value` parameter.

    """
    if not np.issubdtype(bitmask.dtype, np.integer):
        raise TypeError("Input 'bitmask' array must be of integer type.")

    ignore_bits = interpret_bits_value(ignore_bits)

    if ignore_bits is None:
        if good_mask_value:
            mask = np.ones_like(bitmask, dtype=dtype)
        else:
            mask = np.zeros_like(bitmask, dtype=dtype)
        return mask

    ignore_bits = ~bitmask.dtype.type(ignore_bits)

    mask = np.empty_like(bitmask, dtype=np.bool_)
    np.bitwise_and(bitmask, ignore_bits, out=mask, casting='unsafe')

    if good_mask_value:
        np.logical_not(mask, out=mask)

    return mask.astype(dtype=dtype, subok=False, copy=False)