def load(filename, *, gzipped=None, byteorder='big'):
    """Load the nbt file at the specified location.

    By default, the function will figure out by itself if the file is
    gzipped before loading it. You can pass a boolean to the `gzipped`
    keyword only argument to specify explicitly whether the file is
    compressed or not. You can also use the `byteorder` keyword only
    argument to specify whether the file is little-endian or big-endian.
    """
    if gzipped is not None:
        return File.load(filename, gzipped, byteorder)

    # if we don't know we read the magic number
    with open(filename, 'rb') as buff:
        magic_number = buff.read(2)
        buff.seek(0)

        if magic_number == b'\x1f\x8b':
            buff = gzip.GzipFile(fileobj=buff)

        return File.from_buffer(buff, byteorder)