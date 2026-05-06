def sbytes2ilines(stream, encoding="utf8", closer=None):
    """
    CONVERT A STREAM (with read() method) OF (ARBITRARY-SIZED) byte BLOCKS
    TO A LINE (CR-DELIMITED) GENERATOR
    """
    def read():
        try:
            while True:
                bytes_ = stream.read(4096)
                if not bytes_:
                    return
                yield bytes_
        except Exception as e:
            Log.error("Problem iterating through stream", cause=e)
        finally:
            try:
                stream.close()
            except Exception:
                pass

            if closer:
                try:
                    closer()
                except Exception:
                    pass

    return ibytes2ilines(read(), encoding=encoding)