def remove_newlines(xml):
    r"""Remove newlines in the xml.

    If the newline separates words in text, then replace with a space instead.

    >>> remove_newlines('<p>para one</p>\n<p>para two</p>')
    '<p>para one</p><p>para two</p>'
    >>> remove_newlines('<p>line one\nline two</p>')
    '<p>line one line two</p>'
    >>> remove_newlines('one\n1')
    'one 1'
    >>> remove_newlines('hey!\nmore text!')
    'hey! more text!'
    """
    # Normalize newlines.
    xml = xml.replace('\r\n', '\n')
    xml = xml.replace('\r', '\n')
    # Remove newlines that don't separate text. The remaining ones do separate text.
    xml = re.sub(r'(?<=[>\s])\n(?=[<\s])', '', xml)
    xml = xml.replace('\n', ' ')
    return xml.strip()