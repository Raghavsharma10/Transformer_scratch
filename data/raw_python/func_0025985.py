def tkreadline(file=None):

    """Read a line from file while running Tk mainloop.

    If the file is not line-buffered then the Tk mainloop will stop
    running after one character is typed.  The function will still work
    but Tk widgets will stop updating.  This should work OK for stdin and
    other line-buffered filehandles.  If file is omitted, reads from
    sys.stdin.

    The file must have a readline method.  If it does not have a fileno
    method (which can happen e.g. for the status line input on the
    graphics window) then the readline method is simply called directly.
    """

    if file is None:
        file = sys.stdin
    if not hasattr(file, "readline"):
        raise TypeError("file must be a filehandle with a readline method")

    # Call tkread now...
    # BUT, if we get in here for something not GUI-related (e.g. terminal-
    # focused code in a sometimes-GUI app) then skip tkread and simply call
    # readline on the input eg. stdin.  Otherwise we'd fail in _TkRead().read()

    try:
        fd = file.fileno()
    except:
        fd = None

    if (fd and capable.OF_GRAPHICS):
        tkread(fd, 0)
        # if EOF was encountered on a tty, avoid reading again because
        # it actually requests more data
        if not select.select([fd],[],[],0)[0]:
            return ''
    return file.readline()