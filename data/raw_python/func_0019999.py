def format_doc(fun):
    """Formats the documentation in a nicer way and for notebook cells."""
    SEPARATOR = '============================='
    func = cvloop.functions.__dict__[fun]

    doc_lines = ['{}'.format(l).strip() for l in func.__doc__.split('\n')]
    if hasattr(func, '__init__'):
        doc_lines.append(SEPARATOR)
        doc_lines += ['{}'.format(l).strip() for l in
                      func.__init__.__doc__.split('\n')]

    mod_lines = []
    argblock = False
    returnblock = False
    for line in doc_lines:
        if line == SEPARATOR:
            mod_lines.append('\n#### `{}.__init__(...)`:\n\n'.format(fun))
        elif 'Args:' in line:
            argblock = True
            if GENERATE_ARGS:
                mod_lines.append('**{}**\n'.format(line))
        elif 'Returns:' in line:
            returnblock = True
            mod_lines.append('\n**{}**'.format(line))
        elif not argblock and not returnblock:
            mod_lines.append('{}\n'.format(line))
        elif argblock and not returnblock and ':' in line:
            if GENERATE_ARGS:
                mod_lines.append('- *{}:* {}\n'.format(
                    *line.split(':')))
        elif returnblock:
            mod_lines.append(line)
        else:
            mod_lines.append('{}\n'.format(line))
    return mod_lines