def clean_docstring(docstring):
    """Dedent docstring, special casing the first line."""
    docstring = docstring.strip()
    if '\n' in docstring:
        # multiline docstring
        if docstring[0].isspace():
            # whole docstring is indented
            return textwrap.dedent(docstring)
        else:
            # first line not indented, rest maybe
            first, _, rest = docstring.partition('\n')
            return first + '\n' + textwrap.dedent(rest)
    return docstring