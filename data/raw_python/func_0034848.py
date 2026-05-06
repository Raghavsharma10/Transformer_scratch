def parse_buffer(stream, separator=None):
    """
    Returns a dictionary of the lines of a stream, an array of rows of the
     stream (split by separator), and an array of the columns of the stream
     (also split by separator)

    :param stream:
    :param separator:
    :return: dict
    """
    rows = []
    lines = []
    for line, row in parse_lines(stream, separator):
        lines.append(line)
        rows.append(row)
    cols = zip(*rows)
    return {
        'rows': rows,
        'lines': lines,
        'cols': cols,
        }