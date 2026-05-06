def Checksum(params, ctxt, scope, stream, coord):
    """
    Runs a simple checksum on a file and returns the result as a int64. The
    algorithm can be one of the following constants:

    CHECKSUM_BYTE - Treats the file as a set of unsigned bytes
    CHECKSUM_SHORT_LE - Treats the file as a set of unsigned little-endian shorts
    CHECKSUM_SHORT_BE - Treats the file as a set of unsigned big-endian shorts
    CHECKSUM_INT_LE - Treats the file as a set of unsigned little-endian ints
    CHECKSUM_INT_BE - Treats the file as a set of unsigned big-endian ints
    CHECKSUM_INT64_LE - Treats the file as a set of unsigned little-endian int64s
    CHECKSUM_INT64_BE - Treats the file as a set of unsigned big-endian int64s
    CHECKSUM_SUM8 - Same as CHECKSUM_BYTE except result output as 8-bits
    CHECKSUM_SUM16 - Same as CHECKSUM_BYTE except result output as 16-bits
    CHECKSUM_SUM32 - Same as CHECKSUM_BYTE except result output as 32-bits
    CHECKSUM_SUM64 - Same as CHECKSUM_BYTE
    CHECKSUM_CRC16
    CHECKSUM_CRCCCITT
    CHECKSUM_CRC32
    CHECKSUM_ADLER32

    If start and size are zero, the algorithm is run on the whole file. If
    they are not zero then the algorithm is run on size bytes starting at
    address start. See the ChecksumAlgBytes and ChecksumAlgStr functions
    to run more complex algorithms. crcPolynomial and crcInitValue
    can be used to set a custom polynomial and initial value for the
    CRC functions. A value of -1 for these parameters uses the default
    values as described in the Check Sum/Hash Algorithms topic. A negative
    number is returned on error.
    """
    checksum_types = {
        0: "CHECKSUM_BYTE", # Treats the file as a set of unsigned bytes
        1: "CHECKSUM_SHORT_LE", # Treats the file as a set of unsigned little-endian shorts
        2: "CHECKSUM_SHORT_BE", # Treats the file as a set of unsigned big-endian shorts
        3: "CHECKSUM_INT_LE", # Treats the file as a set of unsigned little-endian ints
        4: "CHECKSUM_INT_BE", # Treats the file as a set of unsigned big-endian ints
        5: "CHECKSUM_INT64_LE", # Treats the file as a set of unsigned little-endian int64s
        6: "CHECKSUM_INT64_BE", # Treats the file as a set of unsigned big-endian int64s
        7: "CHECKSUM_SUM8", # Same as CHECKSUM_BYTE except result output as 8-bits
        8: "CHECKSUM_SUM16", # Same as CHECKSUM_BYTE except result output as 16-bits
        9: "CHECKSUM_SUM32", # Same as CHECKSUM_BYTE except result output as 32-bits
        10: "CHECKSUM_SUM64", # Same as CHECKSUM_BYTE
        11: "CHECKSUM_CRC16",
        12: "CHECKSUM_CRCCCITT",
        13: _crc32,
        14: _checksum_Adler32
    }

    if len(params) < 1:
        raise errors.InvalidArguments(coord, "at least 1 argument", "{} args".format(len(params)))
    
    alg = PYVAL(params[0])
    if alg not in checksum_types:
        raise errors.InvalidArguments(coord, "checksum alg must be one of (0-14)", "{}".format(alg))
    
    start = 0
    if len(params) > 1:
        start = PYVAL(params[1])
    
    size = 0
    if len(params) > 2:
        size = PYVAL(params[2])
    
    crc_poly = -1
    if len(params) > 3:
        crc_poly = PYVAL(params[3])
    
    crc_init = -1
    if len(params) > 4:
        crc_init = PYVAL(params[4])
    
    stream_pos = stream.tell()

    if start + size == 0:
        stream.seek(0, 0)
        data = stream.read()
    else:
        stream.seek(start, 0)
        data = stream.read(size)
    
    try:
        return checksum_types[alg](data, crc_init, crc_poly)
    
    finally:
        # yes, this does execute even though a return statement
        # exists within the try
        stream.seek(stream_pos, 0)