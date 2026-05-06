def markup_line(text, offset, marker='>>!<<'):
    """Insert `marker` at `offset` into `text`, and return the marked
    line.

    .. code-block:: python

       >>> markup_line('0\\n1234\\n56', 3)
       1>>!<<234

    """

    begin = text.rfind('\n', 0, offset)
    begin += 1

    end = text.find('\n', offset)

    if end == -1:
        end = len(text)

    return text[begin:offset] + marker + text[offset:end]