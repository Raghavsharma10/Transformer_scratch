def bitfield_to_boolean_mask(bitfield, ignore_flags=0, flip_bits=None,
                             good_mask_value=True, dtype=np.bool_):
    """
    bitfield_to_boolean_mask(bitfield, ignore_flags=None, flip_bits=None, \
good_mask_value=True, dtype=numpy.bool\_)
    Converts an array of bit fields to a boolean (or integer) mask array
    according to a bitmask constructed from the supplied bit flags (see
    ``ignore_flags`` parameter).

    This function is particularly useful to convert data quality arrays to
    boolean masks with selective filtering of DQ flags.

    Parameters
    ----------
    bitfield : numpy.ndarray
        An array of bit flags. By default, values different from zero are
        interpreted as "bad" values and values equal to zero are considered
        as "good" values. However, see ``ignore_flags`` parameter on how to
        selectively ignore some bits in the ``bitfield`` array data.

    ignore_flags : int, str, list, None (Default = 0)
        An integer bitmask, a Python list of bit flags, a comma- or
        '+'-separated string list of integer bit flags that indicate what
        bits in the input ``bitfield`` should be *ignored* (i.e., zeroed), or
        `None`.

        | Setting ``ignore_flags`` to `None` effectively will make
          `bitfield_to_boolean_mask` interpret all ``bitfield`` elements
          as "good" regardless of their value.

        | When ``ignore_flags`` argument is an integer bitmask, it will be
          combined using bitwise-NOT and bitwise-AND with each element of the
          input ``bitfield`` array (``~ignore_flags & bitfield``). If the
          resultant bitfield element is non-zero, that element will be
          interpreted as a "bad" in the output boolean mask and it will be
          interpreted as "good" otherwise. ``flip_bits`` parameter may be used
          to flip the bits (``bitwise-NOT``) of the bitmask thus effectively
          changing the meaning of the ``ignore_flags`` parameter from "ignore"
          to "use only" these flags.

        .. note::

            Setting ``ignore_flags`` to 0 effectively will assume that all
            non-zero elements in the input ``bitfield`` array are to be
            interpreted as "bad".

        | When ``ignore_flags`` argument is an Python list of integer bit
          flags, these flags are added together to create an integer bitmask.
          Each item in the list must be a flag, i.e., an integer that is an
          integer power of 2. In order to flip the bits of the resultant
          bitmask, use ``flip_bits`` parameter.

        | Alternatively, ``ignore_flags`` may be a string of comma- or
          '+'-separated list of integer bit flags that should be added together
          to create an integer bitmask. For example, both ``'4,8'`` and
          ``'4+8'`` are equivalent and indicate that bit flags 4 and 8 in
          the input ``bitfield`` array should be ignored when generating
          boolean mask.

        .. note::

            ``'None'``, ``'INDEF'``, and empty (or all white space) strings
            are special values of string ``ignore_flags`` that are
            interpreted as `None`.

        .. note::

            Each item in the list must be a flag, i.e., an integer that is an
            integer power of 2. In addition, for convenience, an arbitrary
            **single** integer is allowed and it will be interpretted as an
            integer bitmask. For example, instead of ``'4,8'`` one could
            simply provide string ``'12'``.

        .. note::

            When ``ignore_flags`` is a `str` and when it is prepended with
            '~', then the meaning of ``ignore_flags`` parameters will be
            reversed: now it will be interpreted as a list of bit flags to be
            *used* (or *not ignored*) when deciding which elements of the
            input ``bitfield`` array are "bad". Following this convention,
            an ``ignore_flags`` string value of ``'~0'`` would be equivalent
            to setting ``ignore_flags=None``.

        .. warning::

            Because prepending '~' to a string ``ignore_flags`` is equivalent
            to setting ``flip_bits`` to `True`, ``flip_bits`` cannot be used
            with string ``ignore_flags`` and it must be set to `None`.

    flip_bits : bool, None (Default = None)
        Specifies whether or not to invert the bits of the bitmask either
        supplied directly through ``ignore_flags`` parameter or built from the
        bit flags passed through ``ignore_flags`` (only when bit flags are
        passed as Python lists of integer bit flags). Occasionally, it may be
        useful to *consider only specific bit flags* in the ``bitfield``
        array when creating a boolean mask as opposite to *ignoring* specific
        bit flags as ``ignore_flags`` behaves by default. This can be achieved
        by inverting/flipping the bits of the bitmask created from
        ``ignore_flags`` flags which effectively changes the meaning of the
        ``ignore_flags`` parameter from "ignore" to "use only" these flags.
        Setting ``flip_bits`` to `None` means that no bit flipping will be
        performed. Bit flipping for string lists of bit flags must be
        specified by prepending '~' to string bit flag lists
        (see documentation for ``ignore_flags`` for more details).

        .. warning::
            This parameter can be set to either `True` or `False` **ONLY** when
            ``ignore_flags`` is either an integer bitmask or a Python
            list of integer bit flags. When ``ignore_flags`` is either
            `None` or a string list of flags, ``flip_bits`` **MUST** be set
            to `None`.

    good_mask_value : int, bool (Default = True)
        This parameter is used to derive the values that will be assigned to
        the elements in the output boolean mask array that correspond to the
        "good" bit fields (that are 0 after zeroing bits specified by
        ``ignore_flags``) in the input ``bitfield`` array. When
        ``good_mask_value`` is non-zero or `True` then values in the output
        boolean mask array corresponding to "good" bit fields in ``bitfield``
        will be `True` (if ``dtype`` is `numpy.bool_`) or 1 (if ``dtype`` is
        of numerical type) and values of corresponding to "bad" flags will be
        `False` (or 0). When ``good_mask_value`` is zero or `False` then the
        values in the output boolean mask array corresponding to "good" bit
        fields in ``bitfield`` will be `False` (if ``dtype`` is `numpy.bool_`)
        or 0 (if ``dtype`` is of numerical type) and values of corresponding
        to "bad" flags will be `True` (or 1).

    dtype : data-type (Default = numpy.bool\_)
        The desired data-type for the output binary mask array.

    Returns
    -------
    mask : numpy.ndarray
        Returns an array of the same dimensionality as the input ``bitfield``
        array whose elements can have two possible values,
        e.g., `True` or `False` (or 1 or 0 for integer ``dtype``) according to
        values of to the input ``bitfield`` elements, ``ignore_flags``
        parameter, and the ``good_mask_value`` parameter.

    Examples
    --------
        >>> from stsci.tools import bitmask
        >>> import numpy as np
        >>> dqbits = np.asarray([[0,0,1,2,0,8,12,0],[10,4,0,0,0,16,6,0]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags=0, dtype=int)
        array([[1, 1, 0, 0, 1, 0, 0, 1],
               [0, 0, 1, 1, 1, 0, 0, 1]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags=0, dtype=bool)
        array([[ True,  True, False, False,  True, False, False,  True],
               [False, False,  True,  True,  True, False, False,  True]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags=6, good_mask_value=0, dtype=int)
        array([[0, 0, 1, 0, 0, 1, 1, 0],
               [1, 0, 0, 0, 0, 1, 0, 0]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags=~6, good_mask_value=0, dtype=int)
        array([[0, 0, 0, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0, 1, 0]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags=6, flip_bits=True, good_mask_value=0, dtype=int)
        array([[0, 0, 0, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0, 1, 0]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags='~(2+4)', good_mask_value=0, dtype=int)
        array([[0, 0, 0, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0, 1, 0]])
        >>> bitmask.bitfield_to_boolean_mask(dqbits, ignore_flags=[2, 4], flip_bits=True, good_mask_value=0, dtype=int)
        array([[0, 0, 0, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0, 1, 0]])

    """
    bitfield = np.asarray(bitfield)
    if not np.issubdtype(bitfield.dtype, np.integer):
        raise TypeError("Input bitfield array must be of integer type.")

    ignore_mask = interpret_bit_flags(ignore_flags, flip_bits=flip_bits)

    if ignore_mask is None:
        if good_mask_value:
            mask = np.ones_like(bitfield, dtype=dtype)
        else:
            mask = np.zeros_like(bitfield, dtype=dtype)
        return mask

    # filter out bits beyond the maximum supported by the data type:
    ignore_mask = ignore_mask & SUPPORTED_FLAGS

    # invert the "ignore" mask:
    ignore_mask = np.bitwise_not(ignore_mask, dtype=bitfield.dtype,
                                 casting='unsafe')

    mask = np.empty_like(bitfield, dtype=np.bool_)
    np.bitwise_and(bitfield, ignore_mask, out=mask, casting='unsafe')

    if good_mask_value:
        np.logical_not(mask, out=mask)

    return mask.astype(dtype=dtype, subok=False, copy=False)