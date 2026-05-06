def bytes2zip(bytes):
    """
    RETURN COMPRESSED BYTES
    """
    if hasattr(bytes, "read"):
        buff = TemporaryFile()
        archive = gzip.GzipFile(fileobj=buff, mode='w')
        for b in bytes:
            archive.write(b)
        archive.close()
        buff.seek(0)
        from pyLibrary.env.big_data import FileString, safe_size
        return FileString(buff)

    buff = BytesIO()
    archive = gzip.GzipFile(fileobj=buff, mode='w')
    archive.write(bytes)
    archive.close()
    return buff.getvalue()