def get_doc_comments(text):
    r"""
    Return a list of all documentation comments in the file text.  Each
    comment is a pair, with the first element being the comment text and
    the second element being the line after it, which may be needed to
    guess function & arguments.

    >>> get_doc_comments(read_file('examples/module.js'))[0][0][:40]
    '/**\n * This is the module documentation.'
    >>> get_doc_comments(read_file('examples/module.js'))[1][0][7:50]
    'This is documentation for the first method.'
    >>> get_doc_comments(read_file('examples/module.js'))[1][1]
    'function the_first_function(arg1, arg2) '
    >>> get_doc_comments(read_file('examples/module.js'))[2][0]
    '/** This is the documentation for the second function. */'

    """
    def make_pair(match):
        comment = match.group()
        try:
            end = text.find('\n', match.end(0)) + 1
            if '@class' not in comment:
                next_line = next(split_delimited('()', '\n', text[end:]))
            else:
                next_line = text[end:text.find('\n', end)]
        except StopIteration:
            next_line = ''
        return (comment, next_line)
    return [make_pair(match) for match in re.finditer('/\*\*(.*?)\*/', 
            text, re.DOTALL)]