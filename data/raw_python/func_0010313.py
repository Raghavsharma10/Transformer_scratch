def _get_decoder_method(stream_type):
    """ A function to get Device Cloud type to python type converter function.

    :param stream_type: The streams data type
    :return: A function that when called with Device Cloud object will return the python
    native type. If there is no function for the given type, or the `stream_type` is `None`
    the returned function will simply return the object unchanged.
    """
    if stream_type is not None:
        return DSTREAM_TYPE_MAP.get(stream_type.upper(), (lambda x: x, lambda x: x))[0]
    else:
        return lambda x: x