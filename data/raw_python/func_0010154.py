def write_pdf_files(args, infilenames, outfilename):
    """Write pdf file(s) to disk using pdfkit.

    Keyword arguments:
    args -- program arguments (dict)
    infilenames -- names of user-inputted and/or downloaded files (list)
    outfilename -- name of output pdf file (str)
    """
    if not outfilename.endswith('.pdf'):
        outfilename = outfilename + '.pdf'
    outfilename = overwrite_file_check(args, outfilename)

    options = {}
    try:
        if args['multiple']:
            # Multiple files are written one at a time, so infilenames will
            # never contain more than one file here
            infilename = infilenames[0]
            if not args['quiet']:
                print('Attempting to write to {0}.'.format(outfilename))
            else:
                options['quiet'] = None

            if args['xpath']:
                # Process HTML with XPath before writing
                html = parse_html(read_files(infilename), args['xpath'])
                if isinstance(html, list):
                    if isinstance(html[0], str):
                        pk.from_string('\n'.join(html), outfilename,
                                       options=options)
                    else:
                        pk.from_string('\n'.join(lh.tostring(x) for x in html),
                                       outfilename, options=options)
                elif isinstance(html, str):
                    pk.from_string(html, outfilename, options=options)
                else:
                    pk.from_string(lh.tostring(html), outfilename,
                                   options=options)
            else:
                pk.from_file(infilename, outfilename, options=options)
        elif args['single']:
            if not args['quiet']:
                print('Attempting to write {0} page(s) to {1}.'
                      .format(len(infilenames), outfilename))
            else:
                options['quiet'] = None

            if args['xpath']:
                # Process HTML with XPath before writing
                html = parse_html(read_files(infilenames), args['xpath'])
                if isinstance(html, list):
                    if isinstance(html[0], str):
                        pk.from_string('\n'.join(html), outfilename,
                                       options=options)
                    else:
                        pk.from_string('\n'.join(lh.tostring(x) for x in html),
                                       outfilename, options=options)
                elif isinstance(html, str):
                    pk.from_string(html, outfilename, options=options)
                else:
                    pk.from_string(lh.tostring(html), outfilename,
                                   options=options)
            else:
                pk.from_file(infilenames, outfilename, options=options)
        return True
    except (OSError, IOError) as err:
        sys.stderr.write('An error occurred while writing {0}:\n{1}'
                         .format(outfilename, str(err)))
        return False