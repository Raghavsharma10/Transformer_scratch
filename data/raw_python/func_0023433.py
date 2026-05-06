def pack_iterable(messages):
    '''Pack an iterable of messages in the TCP protocol format'''
    # [ 4-byte body size ]
    # [ 4-byte num messages ]
    # [ 4-byte message #1 size ][ N-byte binary data ]
    #      ... (repeated <num_messages> times)
    return pack_string(
        struct.pack('>l', len(messages)) +
        ''.join(map(pack_string, messages)))