def _get_encoder_method(stream_type):
    """A function to get the python type to device cloud type converter function.

    :param stream_type: The streams data type
    :return: A function that when called with the python object will return the serializable
    type for sending to the cloud. If there is no function for the given type, or the `stream_type`
    is `None` the returned function will simply return the object unchanged.
    """
    if stream_type is not None:
        return DSTREAM_TYPE_MAP.get(stream_type.upper(), (lambda x: x, lambda x: x))[1]
    else:
        return lambda x: x