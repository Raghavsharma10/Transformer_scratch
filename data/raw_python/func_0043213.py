def metadata_to_buffers(metadata):
    """
    Transform a dict of metadata into a sequence of buffers.

    :param metadata: The metadata, as a dict.
    :returns: A list of buffers.
    """
    results = []

    for key, value in metadata.items():
        assert len(key) < 256
        assert len(value) < 2 ** 32
        results.extend([
            struct.pack('!B', len(key)),
            key,
            struct.pack('!I', len(value)),
            value,
        ])

    return results