def additive_noise(stream, key='X', scale=1e-1):
    '''Add noise to a data stream.

    Parameters
    ----------
    stream : iterable
        A stream that yields data objects.

    key : string, default='X'
        Name of the field to add noise.

    scale : float, default=0.1
        Scale factor for gaussian noise.

    Yields
    ------
    data : dict
        Updated data objects in the stream.
    '''
    for data in stream:
        noise_shape = data[key].shape
        noise = scale * np.random.randn(*noise_shape)
        data[key] = data[key] + noise
        yield data