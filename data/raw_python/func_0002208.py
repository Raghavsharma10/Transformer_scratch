def datacol_to_array(datacol):
    """Convert DataCol from NCStream v2 into an array with appropriate type.

    Depending on the data type specified, this extracts data from the appropriate members
    and packs into a :class:`numpy.ndarray`, recursing as necessary for compound data types.

    Parameters
    ----------
    datacol : DataCol

    Returns
    -------
    ndarray
        array containing extracted data

    """
    if datacol.dataType == stream.STRING:
        arr = np.array(datacol.stringdata, dtype=np.object)
    elif datacol.dataType == stream.OPAQUE:
        arr = np.array(datacol.opaquedata, dtype=np.object)
    elif datacol.dataType == stream.STRUCTURE:
        members = OrderedDict((mem.name, datacol_to_array(mem))
                              for mem in datacol.structdata.memberData)
        log.debug('Struct members:\n%s', str(members))

        # str() around name necessary because protobuf gives unicode names, but dtype doesn't
        # support them on Python 2
        dt = np.dtype([(str(name), arr.dtype) for name, arr in members.items()])
        log.debug('Struct dtype: %s', str(dt))

        arr = np.empty((datacol.nelems,), dtype=dt)
        for name, arr_data in members.items():
            arr[name] = arr_data
    else:
        # Make an appropriate datatype
        endian = '>' if datacol.bigend else '<'
        dt = data_type_to_numpy(datacol.dataType).newbyteorder(endian)

        # Turn bytes into an array
        arr = np.frombuffer(datacol.primdata, dtype=dt)
        if arr.size != datacol.nelems:
            log.warning('Array size %d does not agree with nelems %d',
                        arr.size, datacol.nelems)
        if datacol.isVlen:
            arr = process_vlen(datacol, arr)
            if arr.dtype == np.object_:
                arr = reshape_array(datacol, arr)
            else:
                # In this case, the array collapsed, need different resize that
                # correctly sizes from elements
                shape = tuple(r.size for r in datacol.section.range) + (datacol.vlens[0],)
                arr = arr.reshape(*shape)
        else:
            arr = reshape_array(datacol, arr)
    return arr