def get_terminal_size(defaultw=80):
    """ Checks various methods to determine the terminal size


    Methods:
    - shutil.get_terminal_size (only Python3)
    - fcntl.ioctl
    - subprocess.check_output
    - os.environ

    Parameters
    ----------
    defaultw : int
        Default width of terminal.


    Returns
    -------
    width, height : int
        Width and height of the terminal. If one of them could not be
        found, None is return in its place.

    """
    if hasattr(shutil_get_terminal_size, "__call__"):
        return shutil_get_terminal_size()
    else:
        try:
            import fcntl, termios, struct
            fd = 0
            hw = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
                                                 '1234'))
            return (hw[1], hw[0])
        except:
            try:
                out = sp.check_output(["tput", "cols"])
                width = int(out.decode("utf-8").strip())
                return (width, None)
            except:
                try:
                    hw = (os.environ['LINES'], os.environ['COLUMNS'])
                    return (hw[1], hw[0])
                except:
                    return (defaultw, None)