def zmq_recv_data(socket, flags=0, copy=True, track=False):
    """Receive data over a socket."""

    data = dict()

    msg = socket.recv_multipart(flags=flags, copy=copy, track=track)

    headers = json.loads(msg[0].decode('ascii'))

    if len(headers) == 0:
        raise StopIteration

    for header, payload in zip(headers, msg[1:]):
        data[header['key']] = np.frombuffer(buffer(payload),
                                            dtype=header['dtype'])
        data[header['key']].shape = header['shape']
        if six.PY2:
            # Legacy python won't let us preserve alignment, skip this step
            continue
        data[header['key']].flags['ALIGNED'] = header['aligned']

    return data