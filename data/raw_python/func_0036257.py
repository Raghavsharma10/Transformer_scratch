def get_terminal_size():
    '''Finds the width of the terminal, or returns a suitable default value.'''
    def read_terminal_size_by_ioctl(fd):
        try:
            import struct, fcntl, termios
            cr = struct.unpack('hh', fcntl.ioctl(1, termios.TIOCGWINSZ,
                                                            '0000'))
        except ImportError:
            return None
        except IOError as e:
            return None
        return cr[1], cr[0]

    cr = read_terminal_size_by_ioctl(0) or \
            read_terminal_size_by_ioctl(1) or \
            read_terminal_size_by_ioctl(2)
    if not cr:
        try:
            import os
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = read_terminal_size_by_ioctl(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        import os
        cr = [80, 25] # 25 rows, 80 columns is the default value
        if os.getenv('ROWS'):
            cr[1] = int(os.getenv('ROWS'))
        if os.getenv('COLUMNS'):
            cr[0] = int(os.getenv('COLUMNS'))

    return cr[1], cr[0]