def buffer_to_metadata(buffer):
    """
    Transform a buffer to a metadata dictionary.

    :param buffer: The buffer, as received in a READY command.
    :returns: A metadata dictionary, with its keys normalized (in
        lowercase).
    """
    offset = 0
    size = len(buffer)
    metadata = {}

    while offset < size:
        name_size = struct.unpack_from('B', buffer, offset)[0]
        offset += 1

        if name_size > size - 4:
            raise ProtocolError(
                "Invalid name size in metadata",
                fatal=True,
            )

        name = buffer[offset:offset + name_size]
        offset += name_size

        value_size = struct.unpack_from('!I', buffer, offset)[0]
        offset += 4

        if value_size > size - name_size - 5:
            raise ProtocolError(
                "Invalid value size in metadata",
                fatal=True,
            )

        value = buffer[offset:offset + value_size]
        offset += value_size
        metadata[name.lower()] = value

    return metadata