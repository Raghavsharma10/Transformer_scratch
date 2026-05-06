def next_line(last_line, next_line_8bit):
    """Compute the next line based on the last line and a 8bit next line.

    The behaviour of the function is specified in :ref:`reqline`.

    :param int last_line: the last line that was processed
    :param int next_line_8bit: the lower 8 bits of the next line
    :return: the next line closest to :paramref:`last_line`

    .. seealso:: :ref:`reqline`
    """
    # compute the line without the lowest byte
    base_line = last_line - (last_line & 255)
    # compute the three different lines
    line = base_line + next_line_8bit
    lower_line = line - 256
    upper_line = line + 256
    # compute the next line
    if last_line - lower_line <= line - last_line:
        return lower_line
    if upper_line - last_line < last_line - line:
        return upper_line
    return line