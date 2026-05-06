def compressed_bytes2ibytes(compressed, size):
    """
    CONVERT AN ARRAY OF BYTES TO A BYTE-BLOCK GENERATOR
    USEFUL IN THE CASE WHEN WE WANT TO LIMIT HOW MUCH WE FEED ANOTHER
    GENERATOR (LIKE A DECOMPRESSOR)
    """
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    for i in range(0, mo_math.ceiling(len(compressed), size), size):
        try:
            block = compressed[i: i + size]
            yield decompressor.decompress(block)
        except Exception as e:
            Log.error("Not expected", e)