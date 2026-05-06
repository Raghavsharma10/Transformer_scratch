def __process_line(line, strip_eol, strip):
    """
    process a single line value.
    """
    if strip:
        line = line.strip()
    elif strip_eol and line.endswith('\n'):
        line = line[:-1]
    return line