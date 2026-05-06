def read_ncstream_data(fobj):
    """Handle reading an NcStream v1 data block from a file-like object."""
    data = read_proto_object(fobj, stream.Data)
    if data.dataType in (stream.STRING, stream.OPAQUE) or data.vdata:
        log.debug('Reading string/opaque/vlen')
        num_obj = read_var_int(fobj)
        log.debug('Num objects: %d', num_obj)
        blocks = [read_block(fobj) for _ in range(num_obj)]
        if data.dataType == stream.STRING:
            blocks = [b.decode('utf-8', errors='ignore') for b in blocks]

        # Again endian isn't coded properly
        dt = data_type_to_numpy(data.dataType).newbyteorder('>')
        if data.vdata:
            return np.array([np.frombuffer(b, dtype=dt) for b in blocks])
        else:
            return np.array(blocks, dtype=dt)
    elif data.dataType in _dtypeLookup:
        log.debug('Reading array data')
        bin_data = read_block(fobj)
        log.debug('Binary data: %s', bin_data)

        # Hard code to big endian for now since it's not encoded correctly
        dt = data_type_to_numpy(data.dataType).newbyteorder('>')

        # Handle decompressing the bytes
        if data.compress == stream.DEFLATE:
            bin_data = zlib.decompress(bin_data)
            assert len(bin_data) == data.uncompressedSize
        elif data.compress != stream.NONE:
            raise NotImplementedError('Compression type {0} not implemented!'.format(
                data.compress))

        # Turn bytes into an array
        return reshape_array(data, np.frombuffer(bin_data, dtype=dt))
    elif data.dataType == stream.STRUCTURE:
        sd = read_proto_object(fobj, stream.StructureData)

        # Make a datatype appropriate to the rows of struct
        endian = '>' if data.bigend else '<'
        dt = np.dtype([(endian, np.void, sd.rowLength)])

        # Turn bytes into an array
        return reshape_array(data, np.frombuffer(sd.data, dtype=dt))
    elif data.dataType == stream.SEQUENCE:
        log.debug('Reading sequence')
        blocks = []
        magic = read_magic(fobj)
        while magic != MAGIC_VEND:
            if magic == MAGIC_VDATA:
                log.error('Bad magic for struct/seq data!')
            blocks.append(read_proto_object(fobj, stream.StructureData))
            magic = read_magic(fobj)
        return data, blocks
    else:
        raise NotImplementedError("Don't know how to handle data type: {0}".format(
            data.dataType))