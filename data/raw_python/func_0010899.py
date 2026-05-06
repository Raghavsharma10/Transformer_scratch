def parse_gzip(file_path):
    """Return a decoded API to the data from a file path. File is gzip compressed.
    :param file_path: the input file path. Data is gzip compressed.
    :return an API to decoded data"""
    newDecoder = MMTFDecoder()
    newDecoder.decode_data(_unpack(gzip.open(file_path, "rb")))
    return newDecoder