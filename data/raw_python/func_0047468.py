def strip_comments(code):
    '''Returns the headers with comments removed.
    '''
    single_comment = compile('//.*')  # single line comment
    multi_comment = compile('/\\*\\*.*?\\*/', re.DOTALL)  # multiline comment
    code = sub(single_comment, '', code)
    code = sub(multi_comment, '', code)
    return '\n'.join([c for c in code.splitlines() if c.strip()])