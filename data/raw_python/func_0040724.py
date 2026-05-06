def get_indented_block(prefix_lines):
    """Returns an integer.

    The return value is the number of lines that belong to block begun
    on the first line.

    Parameters
    ----------

      prefix_lines : list of basestring pairs
        Each pair corresponds to a line of SHPAML source code. The
        first element of each pair is indentation. The second is the
        remaining part of the line, except for trailing newline.
    """
    prefix, line = prefix_lines[0]
    len_prefix = len(prefix)

    # Find the first nonempty line with len(prefix) <= len(prefix)
    i = 1
    while i < len(prefix_lines):
        new_prefix, line = prefix_lines[i]
        if line and len(new_prefix) <= len_prefix:
            break
        i += 1

    # Rewind to exclude empty lines
    while i - 1 > 0 and prefix_lines[i - 1][1] == '':
        i -= 1

    return i