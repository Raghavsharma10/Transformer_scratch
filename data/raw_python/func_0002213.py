def unpack_attribute(att):
    """Unpack an embedded attribute into a python or numpy object."""
    if att.unsigned:
        log.warning('Unsupported unsigned attribute!')

    # TDS 5.0 now has a dataType attribute that takes precedence
    if att.len == 0:  # Empty
        val = None
    elif att.dataType == stream.STRING:  # Then look for new datatype string
        val = att.sdata
    elif att.dataType:  # Then a non-zero new data type
        val = np.frombuffer(att.data,
                            dtype='>' + _dtypeLookup[att.dataType], count=att.len)
    elif att.type:  # Then non-zero old-data type0
        val = np.frombuffer(att.data,
                            dtype=_attrConverters[att.type], count=att.len)
    elif att.sdata:  # This leaves both 0, try old string
        val = att.sdata
    else:  # Assume new datatype is Char (0)
        val = np.array(att.data, dtype=_dtypeLookup[att.dataType])

    if att.len == 1:
        val = val[0]

    return att.name, val