def _wrap_stream(stream):
        """Returns a TextIOWrapper around the given stream that handles UTF-8
        encoding/decoding.
        """
        if hasattr(stream, "buffer"):
            return io.TextIOWrapper(stream.buffer, encoding="utf-8")
        elif hasattr(stream, "readable"):
            return io.TextIOWrapper(stream, encoding="utf-8")
        # Python 2.x stdin and stdout are just files
        else:
            return io.open(stream.fileno(), mode=stream.mode, encoding="utf-8")