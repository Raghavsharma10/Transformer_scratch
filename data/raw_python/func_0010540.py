def writeToDelimitedString(obj, stream=None):
    """
    Stanford CoreNLP uses the Java "writeDelimitedTo" function, which
    writes the size (and offset) of the buffer before writing the object.
    This function handles parsing this message starting from offset 0.

    @returns how many bytes of @buf were consumed.
    """
    if stream is None:
        stream = BytesIO()

    _EncodeVarint(stream.write, obj.ByteSize(), True)
    stream.write(obj.SerializeToString())
    return stream