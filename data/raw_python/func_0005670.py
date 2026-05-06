def tracebacks_from_lines(lines_iter):
    """Generator that yields tracebacks found in a lines iterator

    The lines iterator can be:

    - a file-like object
    - a list (or deque) of lines.
    - any other iterable sequence of strings
    """

    tbgrep = TracebackGrep()

    for line in lines_iter:
        tb = tbgrep.process(line)
        if tb:
            yield tb