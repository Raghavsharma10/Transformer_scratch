def scompressed2ibytes(stream):
    """
    :param stream:  SOMETHING WITH read() METHOD TO GET MORE BYTES
    :return: GENERATOR OF UNCOMPRESSED BYTES
    """
    def more():
        try:
            while True:
                bytes_ = stream.read(4096)
                if not bytes_:
                    return
                yield bytes_
        except Exception as e:
            Log.error("Problem iterating through stream", cause=e)
        finally:
            with suppress_exception:
                stream.close()

    return icompressed2ibytes(more())