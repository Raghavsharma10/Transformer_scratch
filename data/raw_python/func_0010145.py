def get_parsed_text(args, infilename):
    """Parse and return text content of infiles.

    Keyword arguments:
    args -- program arguments (dict)
    infilenames -- name of user-inputted and/or downloaded file (str)

    Return a list of strings of text.
    """
    parsed_text = []
    if infilename.endswith('.html'):
        # Convert HTML to lxml object for content parsing
        html = lh.fromstring(read_files(infilename))
        text = None
    else:
        html = None
        text = read_files(infilename)

    if html is not None:
        parsed_text = parse_text(html, args['xpath'], args['filter'],
                                 args['attributes'])
    elif text is not None:
        parsed_text = parse_text(text, args['xpath'], args['filter'])
    else:
        if not args['quiet']:
            sys.stderr.write('Failed to parse text from {0}.\n'
                             .format(infilename))
    return parsed_text