def read_messages(fobj, magic_table):
    """Read messages from a file-like object until stream is exhausted."""
    messages = []

    while True:
        magic = read_magic(fobj)
        if not magic:
            break

        func = magic_table.get(magic)
        if func is not None:
            messages.append(func(fobj))
        else:
            log.error('Unknown magic: ' + str(' '.join('{0:02x}'.format(b)
                                                       for b in bytearray(magic))))

    return messages