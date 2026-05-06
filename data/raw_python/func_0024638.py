def get_terminal_size(fallback=(80, 24)):
    """
    Return tuple containing columns and rows of controlling terminal, trying harder
    than shutil.get_terminal_size to find a tty before returning fallback.

    Theoretically, stdout, stderr, and stdin could all be different ttys that could
    cause us to get the wrong measurements (instead of using the fallback) but the much more
    common case is that IO is piped.
    """
    for stream in [sys.__stdout__, sys.__stderr__, sys.__stdin__]:
        try:
            # Make WINSIZE call to terminal
            data = fcntl.ioctl(stream.fileno(), TIOCGWINSZ, b"\x00\x00\00\x00")
        except OSError:
            pass
        else:
            # Unpack two shorts from ioctl call
            lines, columns = struct.unpack("hh", data)
            break
    else:
        columns, lines = fallback

    return columns, lines