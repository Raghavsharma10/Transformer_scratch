def header(*msg, level='h1', separator=" ", print_out=print):
    ''' Print header block in text mode
    '''
    out_string = separator.join(str(x) for x in msg)
    if level == 'h0':
        # box_len = 80 if len(msg) < 80 else len(msg)
        box_len = 80
        print_out('+' + '-' * (box_len + 2))
        print_out("| %s" % out_string)
        print_out('+' + '-' * (box_len + 2))
    elif level == 'h1':
        print_out("")
        print_out(out_string)
        print_out('-' * 60)
    elif level == 'h2':
        print_out('\t%s' % out_string)
        print_out('\t' + ('-' * 40))
    else:
        print_out('\t\t%s' % out_string)
        print_out('\t\t' + ('-' * 20))