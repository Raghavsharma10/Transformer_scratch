def interpretDQvalue(input):
    """
    Converts an integer 'input' into its component bit values as a list of
    power of 2 integers.

    For example, the bit value 1027 would return [1, 2, 1024]
    """

    nbits = 16
    # We will only support integer values up to 2**128
    for iexp in [16, 32, 64, 128]:
        # Find out whether the input value is less than 2**iexp
        if (input // (2 ** iexp)) == 0:
            # when it finally is, we have identified how many bits can be used to
            # describe this input bitvalue
            nbits = iexp
            break

    # Find out how 'dtype' values are described on this machine
    a = np.zeros(1, dtype='int16')
    atype_descr = a.dtype.descr[0][1]
    # Use this description to build the description we need for our input integer
    dtype_str = atype_descr[:2] + str(nbits // 8)
    result = np.zeros(nbits + 1, dtype=dtype_str)

    # For each bit, determine whether it has been set in the input value or not
    for n in range(nbits + 1):
        i = 2 ** n
        if input & i > 0:
            # record which bit has been set as the power-of-2 integer
            result[n] = i

    # Return the non-zero unique values as a Python list
    return np.delete(np.unique(result), 0).tolist()