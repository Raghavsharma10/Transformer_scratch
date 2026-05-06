def insert_string_at_line(input_file: str,
                          string_to_be_inserted: str,
                          put_at_line_number: int,
                          output_file: str,
                          append: bool = True,
                          newline_character: str = '\n'):
    r"""Write a string at the specified line.

    :parameter input_file: the file that needs to be read.
    :parameter string_to_be_inserted: the string that needs to be added.
    :parameter put_at_line_number: the line number on which to append the
         string.
    :parameter output_file: the file that needs to be written with the new
         content.
    :parameter append: decides whether to append or prepend the string at the
         selected line. Defaults to ``True``.
    :parameter newline_character: set the character used to fill the file
         in case line_number is greater than the number of lines of
         input_file. Defaults to ``\n``.
    :type input_file: str
    :type string_to_be_inserted: str
    :type line_number: int
    :type output_file: str
    :type append: bool
    :type newline_character: str
    :returns: None
    :raises: LineOutOfFileBoundsError or a built-in exception.

    .. note::
         Line numbers start from ``1``.
    """
    assert put_at_line_number >= 1

    with open(input_file, 'r') as f:
        lines = f.readlines()

    line_counter = 1
    i = 0
    loop = True
    extra_lines_done = False
    line_number_after_eof = len(lines) + 1
    with atomic_write(output_file, overwrite=True) as f:
        while loop:
            if put_at_line_number > len(
                    lines) and line_counter == line_number_after_eof:
                # There are extra lines to write.
                line = str()
            else:
                line = lines[i]
            # It is ok if the position of line to be written is greater
            # than the last line number of the input file. We just need to add
            # the appropriate number of new line characters which will fill
            # the non existing lines of the output file.
            if put_at_line_number > len(
                    lines) and line_counter == line_number_after_eof:
                for additional_newlines in range(
                        0, put_at_line_number - len(lines) - 1):
                    # Skip the newline in the line where we need to insert
                    # the new string.
                    f.write(newline_character)
                    line_counter += 1
                    i += 1
                extra_lines_done = True

            if line_counter == put_at_line_number:
                # A very simple append operation: if the original line ends
                # with a '\n' character, the string will be added on the next
                # line...
                if append:
                    line = line + string_to_be_inserted
                # ...otherwise the string is prepended.
                else:
                    line = string_to_be_inserted + line
            f.write(line)
            line_counter += 1
            i += 1
            # Quit the loop if there is nothing more to write.
            if i >= len(lines):
                loop = False
            # Continue looping if there are still extra lines to write.
            if put_at_line_number > len(lines) and not extra_lines_done:
                loop = True