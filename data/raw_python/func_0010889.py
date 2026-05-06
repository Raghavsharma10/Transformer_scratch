def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num

    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    dt = numpy.dtype('>i' + str(num))
    return numpy.frombuffer(in_bytes, dt)