def safe_size(source):
    """
    READ THE source UP TO SOME LIMIT, THEN COPY TO A FILE IF TOO BIG
    RETURN A str() OR A FileString()
    """

    if source is None:
        return None

    total_bytes = 0
    bytes = []
    b = source.read(MIN_READ_SIZE)
    while b:
        total_bytes += len(b)
        bytes.append(b)
        if total_bytes > MAX_STRING_SIZE:
            try:
                data = FileString(TemporaryFile())
                for bb in bytes:
                    data.write(bb)
                del bytes
                del bb
                b = source.read(MIN_READ_SIZE)
                while b:
                    total_bytes += len(b)
                    data.write(b)
                    b = source.read(MIN_READ_SIZE)
                data.seek(0)
                Log.note("Using file of size {{length}} instead of str()",  length= total_bytes)

                return data
            except Exception as e:
                Log.error("Could not write file > {{num}} bytes",  num= total_bytes, cause=e)
        b = source.read(MIN_READ_SIZE)

    data = b"".join(bytes)
    del bytes
    return data