def first_paragraph(multiline_str, without_trailing_dot=True, maxlength=None):
    '''Return first paragraph of multiline_str as a oneliner.

    When without_trailing_dot is True, the last char of the first paragraph
    will be removed, if it is a dot ('.').

    Examples:
        >>> multiline_str = 'first line\\nsecond line\\n\\nnext paragraph'
        >>> print(first_paragraph(multiline_str))
        first line second line

        >>> multiline_str = 'first \\n second \\n  \\n next paragraph '
        >>> print(first_paragraph(multiline_str))
        first second

        >>> multiline_str = 'first line\\nsecond line\\n\\nnext paragraph'
        >>> print(first_paragraph(multiline_str, maxlength=3))
        fir

        >>> multiline_str = 'first line\\nsecond line\\n\\nnext paragraph'
        >>> print(first_paragraph(multiline_str, maxlength=78))
        first line second line

        >>> multiline_str = 'first line.'
        >>> print(first_paragraph(multiline_str))
        first line

        >>> multiline_str = 'first line.'
        >>> print(first_paragraph(multiline_str, without_trailing_dot=False))
        first line.

        >>> multiline_str = ''
        >>> print(first_paragraph(multiline_str))
        <BLANKLINE>
    '''
    stripped = '\n'.join([line.strip() for line in multiline_str.splitlines()])
    paragraph = stripped.split('\n\n')[0]
    res = paragraph.replace('\n', ' ')
    if without_trailing_dot:
        res = res.rsplit('.', 1)[0]
    if maxlength:
        res = res[0:maxlength]
    return res