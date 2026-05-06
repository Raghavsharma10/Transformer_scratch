def tracebacks_from_file(fileobj, reverse=False):
    """Generator that yields tracebacks found in a file object

    With reverse=True, searches backwards from the end of the file.
    """

    if reverse:
        lines = deque()

        for line in BackwardsReader(fileobj):
            lines.appendleft(line)
            if tb_head in line:
                yield next(tracebacks_from_lines(lines))
                lines.clear()
    else:
        for traceback in tracebacks_from_lines(fileobj):
            yield traceback