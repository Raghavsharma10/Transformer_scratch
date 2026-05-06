def table(text):
    """Format the text as a table.

    Text in format:

    first | second
    row 2 col 1 | 4

    Will be formatted as::

        +-------------+--------+
        | first       | second |
        +-------------+--------+
        | row 2 col 1 | 4      |
        +-------------+--------+

    Args:
        text (str): Text that needs to be formatted.

    Returns:
        str: Formatted string.
    """

    def table_bar(col_lengths):
        return "+-%s-+%s" % (
            "-+-".join(["-" * length for length in col_lengths]),
            os.linesep,
        )

    rows = []
    for line in text.splitlines():
        rows.append([part.strip() for part in line.split("|")])
    max_cols = max(map(len, rows))
    col_lengths = [0] * max_cols
    for row in rows:
        cols = len(row)
        if cols < max_cols:
            row.extend([""] * (max_cols - cols))
        for i, col in enumerate(row):
            col_length = len(col)
            if col_length > col_lengths[i]:
                col_lengths[i] = col_length
    text = table_bar(col_lengths)
    for i, row in enumerate(rows):
        cols = []
        for i, col in enumerate(row):
            cols.append(col.ljust(col_lengths[i]))
        text += "| %s |%s" % (" | ".join(cols), os.linesep)
        text += table_bar(col_lengths)
    return text