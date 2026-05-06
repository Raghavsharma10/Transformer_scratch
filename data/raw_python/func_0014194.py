def crc32(filename):
    '''
    Calculates the CRC checksum for a file.
    Using CRC32 because security isn't the issue and don't need perfect noncollisions.
    We just need to know if a file has changed.

    On my machine, crc32 was 20 times faster than any hashlib algorithm,
    including blake and md5 algorithms.
    '''
    result = 0
    with open(filename, 'rb') as fin:
        while True:
            chunk = fin.read(48)
            if len(chunk) == 0:
                break
            result = zlib.crc32(chunk, result)
    return result