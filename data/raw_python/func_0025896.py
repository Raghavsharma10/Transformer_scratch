def interpret_bit_flags(bit_flags, flip_bits=None):
    """
    Converts input bit flags to a single integer value (bitmask) or `None`.

    When input is a list of flags (either a Python list of integer flags or a
    sting of comma- or '+'-separated list of flags), the returned bitmask
    is obtained by summing input flags.

    .. note::
        In order to flip the bits of the returned bitmask,
        for input of `str` type, prepend '~' to the input string. '~' must
        be prepended to the *entire string* and not to each bit flag! For
        input that is already a bitmask or a Python list of bit flags, set
        `flip_bits` for `True` in order to flip the bits of the returned
        bitmask.

    Parameters
    ----------
    bit_flags : int, str, list, None
        An integer bitmask or flag, `None`, a string of comma- or
        '+'-separated list of integer bit flags, or a Python list of integer
        bit flags. If `bit_flags` is a `str` and if it is prepended with '~',
        then the output bitmask will have its bits flipped (compared to simple
        sum of input flags). For input `bit_flags` that is already a bitmask
        or a Python list of bit flags, bit-flipping can be controlled through
        `flip_bits` parameter.

    flip_bits : bool, None
        Indicates whether or not to flip the bits of the returned bitmask
        obtained from input bit flags. This parameter must be set to `None`
        when input `bit_flags` is either `None` or a Python list of flags.

    Returns
    -------
    bitmask : int or None
        Returns and integer bit mask formed from the input bit value
        or `None` if input `bit_flags` parameter is `None` or an empty string.
        If input string value was prepended with '~' (or `flip_bits` was
        set to `True`), then returned value will have its bits flipped
        (inverse mask).

    Examples
    --------
        >>> from stsci.tools.bitmask import interpret_bit_flags
        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags(28))
        '0000000000011100'
        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags('4,8,16'))
        '0000000000011100'
        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags('~4,8,16'))
        '1111111111100011'
        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags('~(4+8+16)'))
        '1111111111100011'
        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags([4, 8, 16]))
        '0000000000011100'
        >>> "{0:016b}".format(0xFFFF & interpret_bit_flags([4, 8, 16], flip_bits=True))
        '1111111111100011'

    """
    has_flip_bits = flip_bits is not None
    flip_bits = bool(flip_bits)
    allow_non_flags = False

    if _is_int(bit_flags):
        return (~int(bit_flags) if flip_bits else int(bit_flags))

    elif bit_flags is None:
        if has_flip_bits:
            raise TypeError(
                "Keyword argument 'flip_bits' must be set to 'None' when "
                "input 'bit_flags' is None."
            )
        return None

    elif isinstance(bit_flags, six.string_types):
        if has_flip_bits:
            raise TypeError(
                "Keyword argument 'flip_bits' is not permitted for "
                "comma-separated string lists of bit flags. Prepend '~' to "
                "the string to indicate bit-flipping."
            )

        bit_flags = str(bit_flags).strip()

        if bit_flags.upper() in ['', 'NONE', 'INDEF']:
            return None

        # check whether bitwise-NOT is present and if it is, check that it is
        # in the first position:
        bitflip_pos = bit_flags.find('~')
        if bitflip_pos == 0:
            flip_bits = True
            bit_flags = bit_flags[1:].lstrip()
        else:
            if bitflip_pos > 0:
                raise ValueError("Bitwise-NOT must precede bit flag list.")
            flip_bits = False

        # basic check for correct use of parenthesis:
        while True:
            nlpar = bit_flags.count('(')
            nrpar = bit_flags.count(')')

            if nlpar == 0 and nrpar == 0:
                break

            if nlpar != nrpar:
                raise ValueError("Unbalanced parantheses in bit flag list.")

            lpar_pos = bit_flags.find('(')
            rpar_pos = bit_flags.rfind(')')
            if lpar_pos > 0 or rpar_pos < (len(bit_flags) - 1):
                raise ValueError("Incorrect syntax (incorrect use of "
                                 "parenthesis) in bit flag list.")

            bit_flags = bit_flags[1:-1].strip()

        if ',' in bit_flags:
            bit_flags = bit_flags.split(',')

        elif '+' in bit_flags:
            bit_flags = bit_flags.split('+')

        else:
            if bit_flags == '':
                raise ValueError(
                    "Empty bit flag lists not allowed when either bitwise-NOT "
                    "or parenthesis are present."
                )
            bit_flags = [bit_flags]

        allow_non_flags = len(bit_flags) == 1

    elif hasattr(bit_flags, '__iter__'):
        if not all([_is_int(flag) for flag in bit_flags]):
            raise TypeError("Each bit flag in a list must be an integer.")

    else:
        raise TypeError("Unsupported type for argument 'bit_flags'.")

    bitset = set(map(int, bit_flags))
    if len(bitset) != len(bit_flags):
        warnings.warn("Duplicate bit flags will be ignored")

    bitmask = 0
    for v in bitset:
        if not is_bit_flag(v) and not allow_non_flags:
            raise ValueError("Input list contains invalid (not powers of two) "
                             "bit flags")
        bitmask += v

    if flip_bits:
        bitmask = ~bitmask

    return bitmask