def lines(text):
    """
    Generator function to yield lines (delimited with ``'\n'``) stored in
    ``text``. This is useful when a regular expression should only match on a
    per line basis in a memory efficient way.
    """
    assert text is not None
    assert '\r' not in text
    previous_newline_index = 0
    newline_index = text.find('\n')
    while newline_index != -1:
        yield text[previous_newline_index:newline_index]
        previous_newline_index = newline_index + 1
        newline_index = text.find('\n', previous_newline_index)
    last_line = text[previous_newline_index:]
    if last_line != '':
        yield last_line