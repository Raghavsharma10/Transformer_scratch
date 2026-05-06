def parse(file_path):
    """Return a decoded API to the data from a file path.

    :param file_path: the input file path. Data is not entropy compressed (e.g. gzip)
    :return an API to decoded data """
    newDecoder = MMTFDecoder()
    with open(file_path, "rb") as fh:
        newDecoder.decode_data(_unpack(fh))
    return newDecoder