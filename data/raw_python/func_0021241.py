def decompose_jamo(compound):
    """Return a tuple of jamo character constituents of a compound.
    Note: Non-compound characters are echoed back.

    WARNING: Archaic jamo compounds will raise NotImplementedError.
    """
    if len(compound) != 1:
        raise TypeError("decompose_jamo() expects a single character,",
                        "but received", type(compound), "length",
                        len(compound))
    if compound not in JAMO_COMPOUNDS:
        # Strict version:
        # raise TypeError("decompose_jamo() expects a compound jamo,",
        #                 "but received", compound)
        return compound
    return _JAMO_TO_COMPONENTS.get(compound, compound)