def md2rst(md_lines):
    'Only converts headers'
    lvl2header_char = {1: '=', 2: '-', 3: '~'}
    for md_line in md_lines:
        if md_line.startswith('#'):
            header_indent, header_text = md_line.split(' ', 1)
            yield header_text
            header_char = lvl2header_char[len(header_indent)]
            yield header_char * len(header_text)
        else:
            yield md_line