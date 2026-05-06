def parseFromDelimitedString(obj, buf, offset=0):
    """
    Stanford CoreNLP uses the Java "writeDelimitedTo" function, which
    writes the size (and offset) of the buffer before writing the object.
    This function handles parsing this message starting from offset 0.

    @returns how many bytes of @buf were consumed.
    """
    size, pos = _DecodeVarint(buf, offset)
    obj.ParseFromString(buf[offset+pos:offset+pos+size])
    return pos+size