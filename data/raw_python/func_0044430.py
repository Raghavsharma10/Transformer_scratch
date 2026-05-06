def binary_stream(stream):
    """Ensure a stream is binary on Windows.

    :return: the stream
    """
    try:
        import os
        if os.name == 'nt':
            fileno = getattr(stream, 'fileno', None)
            if fileno:
                no = fileno()
                if no >= 0:     # -1 means we're working as subprocess
                    import msvcrt
                    msvcrt.setmode(no, os.O_BINARY)
    except ImportError:
        pass
    return stream