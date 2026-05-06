def BackwardsReader(file, BLKSIZE = 4096):
    """Read a file line by line, backwards"""

    buf = ""

    file.seek(0, 2)
    lastchar = file.read(1)
    trailing_newline = (lastchar == "\n")

    while 1:
        newline_pos = buf.rfind("\n")
        pos = file.tell()
        if newline_pos != -1:
            # Found a newline
            line = buf[newline_pos+1:]
            buf = buf[:newline_pos]
            if pos or newline_pos or trailing_newline:
                line += "\n"
            yield line
        elif pos:
            # Need to fill buffer
            toread = min(BLKSIZE, pos)
            file.seek(pos-toread, 0)
            buf = file.read(toread) + buf
            file.seek(pos-toread, 0)
            if pos == toread:
                buf = "\n" + buf
        else:
            # Start-of-file
            return