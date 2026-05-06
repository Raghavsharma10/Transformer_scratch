def glBufferData(target, data, usage):
    """ Data can be numpy array or the size of data to allocate.
    """
    if isinstance(data, int):
        size = data
        data = None
    else:
        size = data.nbytes
    GL.glBufferData(target, size, data, usage)