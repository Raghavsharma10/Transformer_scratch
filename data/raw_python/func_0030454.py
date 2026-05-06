def drop_empty(rows):
    """Transpose the columns into rows, remove all of the rows that are empty after the first cell, then
    transpose back. The result is that columns that have a header but no data in the body are removed, assuming
    the header is the first row. """
    return zip(*[col for col in zip(*rows) if bool(filter(bool, col[1:]))])