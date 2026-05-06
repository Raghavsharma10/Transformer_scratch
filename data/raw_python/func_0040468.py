def remove_line_interval(input_file: str, delete_line_from: int,
                         delete_line_to: int, output_file: str):
    r"""Remove a line interval.

    :parameter input_file: the file that needs to be read.
    :parameter delete_line_from: the line number from which start deleting.
    :parameter delete_line_to: the line number to which stop deleting.
    :parameter output_file: the file that needs to be written without the
         selected lines.
    :type input_file: str
    :type delete_line_from: int
    :type delete_line_to: int
    :type output_file: str
    :returns: None
    :raises: LineOutOfFileBoundsError or a built-in exception.

    .. note::
         Line numbers start from ``1``.

    .. note::
         It is possible to remove a single line only. This happens when
         the parameters delete_line_from and delete_line_to are equal.
    """
    assert delete_line_from >= 1
    assert delete_line_to >= 1

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Invalid line ranges.
    # Base case delete_line_to - delete_line_from == 0: single line.
    if delete_line_to - delete_line_from < 0:
        raise NegativeLineRangeError
    if delete_line_from > len(lines) or delete_line_to > len(lines):
        raise LineOutOfFileBoundsError

    line_counter = 1
    # Rewrite the file without the string.
    with atomic_write(output_file, overwrite=True) as f:
        for line in lines:
            # Ignore the line interval where the content to be deleted lies.
            if line_counter >= delete_line_from and line_counter <= delete_line_to:
                pass
            # Write the rest of the file.
            else:
                f.write(line)
            line_counter += 1