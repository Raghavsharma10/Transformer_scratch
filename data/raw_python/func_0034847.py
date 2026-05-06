def parse_lines(stream, separator=None):
    """
    Takes each line of a stream, creating a generator that yields
    tuples of line, row - where row is the line split by separator
    (or by whitespace if separator is None.

    :param stream:
    :param separator: (optional)
    :return: generator
    """
    separator = None if separator is None else unicode(separator)
    for line in stream:
        line = line.rstrip(u'\r\n')
        row = [interpret_segment(i) for i in line.split(separator)]
        yield line, row