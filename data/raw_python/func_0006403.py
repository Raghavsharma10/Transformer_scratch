def send_meta_data(socket, conf, name):
    '''Sends the config via ZeroMQ to a specified socket. Is called at the beginning of a run and when the config changes. Conf can be any config dictionary.
    '''
    meta_data = dict(
        name=name,
        conf=conf
    )
    try:
        socket.send_json(meta_data, flags=zmq.NOBLOCK)
    except zmq.Again:
        pass